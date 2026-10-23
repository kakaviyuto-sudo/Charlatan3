from __future__ import annotations

import io
import os
import queue
import tempfile
import threading
import time
import wave
from typing import Any, Optional

import numpy as np

from core.logger import get_logger
from services.base_service import BaseService

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None


class VoiceLoopService(BaseService):
    """Background loop that continuously consumes audio_queue and turns it into text + LLM reply."""

    name = "voice_loop"

    def __init__(
        self,
        runtime: Any,
        *,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "ja",
        llm_service_name: str = "llm",
    ):
        super().__init__(runtime)
        self.logger = get_logger("services.voice_loop")
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.llm_service_name = llm_service_name
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.whisper_model = self._load_whisper_model()

    def _load_whisper_model(self):
        if WhisperModel is None:
            self.logger.warning("faster-whisper is not installed; voice loop will remain idle until installed.")
            return None
        try:
            return WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Failed to initialize WhisperModel: %s", exc)
            return None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker, name="VoiceLoopWorker", daemon=True)
        self.thread.start()
        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
        self.state.set_service_status(self.name, "stopped")

    def _write_wav_temp(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio_i16 = np.clip(audio, -1.0, 1.0) * 32767
        audio_i16 = audio_i16.astype("<i2")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_i16.tobytes())
        return path

    def _transcribe(self, audio: np.ndarray) -> str:
        if self.whisper_model is None:
            raise RuntimeError("faster-whisper is not available")

        path = self._write_wav_temp(audio)
        try:
            segments, _info = self.whisper_model.transcribe(path, language=self.language)
            text = " ".join(seg.text for seg in segments).strip()
            return text
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _get_llm_service(self):
        getter = getattr(self.state, "get_service", None)
        if getter is not None:
            return getter(self.llm_service_name)
        return self.state.services.get(self.llm_service_name)

    def _push_text(self, text: str) -> None:
        if not text:
            return

        if self.state.recognized_text_queue is None:
            self.state.recognized_text_queue = []

        if hasattr(self.state.recognized_text_queue, "append"):
            self.state.recognized_text_queue.append(text)
        else:
            try:
                self.state.recognized_text_queue.put_nowait(text)
            except Exception:
                pass

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            audio_queue = getattr(self.state, "audio_queue", None)
            if audio_queue is None:
                time.sleep(0.2)
                continue

            try:
                audio = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                time.sleep(0.2)
                continue

            try:
                transcript = self._transcribe(audio)
                if not transcript:
                    continue

                self.logger.info("Recognized: %s", transcript)
                self._push_text(transcript)

                llm = self._get_llm_service()
                if llm is not None:
                    reply = llm.generate_simple(transcript)
                    self.logger.info("LLM reply: %s", reply)
                    if hasattr(self.state, "history"):
                        self.state.history.append({"role": "user", "content": transcript})
                        self.state.history.append({"role": "assistant", "content": reply})

            except Exception as exc:
                self.logger.exception("Voice loop processing failed: %s", exc)


__all__ = ["VoiceLoopService"]
