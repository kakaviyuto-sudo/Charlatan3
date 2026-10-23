from __future__ import annotations

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
    """Background loop: audio_queue -> Whisper -> LLM response.

    Features:
    - ignores very short false positives
    - supports wake word trigger
    - appends transcript + reply into runtime state history
    """

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
        min_transcript_chars: int = 3,
        wake_word: Optional[str] = None,
        wake_word_mode: str = "enabled",
        ignore_short_phrases: tuple[str, ...] = ("あっ", "ああ", "ええ", "はい", "うん"),
    ):
        super().__init__(runtime)
        self.logger = get_logger("services.voice_loop")
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.llm_service_name = llm_service_name
        self.min_transcript_chars = min_transcript_chars
        self.wake_word = wake_word
        self.wake_word_mode = wake_word_mode
        self.ignore_short_phrases = ignore_short_phrases

        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.whisper_model = self._load_whisper_model()

    def _load_whisper_model(self):
        if WhisperModel is None:
            self.logger.warning("faster-whisper is not installed; voice loop idle.")
            return None
        try:
            return WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
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

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.strip().split())

    def _is_ignored_short_phrase(self, text: str) -> bool:
        if not text:
            return True
        canon = self._normalize_text(text)
        if len(canon) < self.min_transcript_chars:
            return True
        if canon in self.ignore_short_phrases:
            return True
        return False

    def _strip_wake_word(self, text: str) -> str:
        if self.wake_word is None:
            return text
        w = self.wake_word
        canon = self._normalize_text(text)
        if canon.startswith(w):
            rest = canon[len(w):].strip(" 、。！？")
            return rest
        return canon

    def _should_process(self, text: str) -> tuple[bool, str]:
        canon = self._normalize_text(text)

        if self._is_ignored_short_phrase(canon):
            return False, ""

        if self.wake_word is None:
            return True, canon

        if not canon.startswith(self.wake_word):
            return False, ""

        stripped = canon[len(self.wake_word):].strip(" 、。！？")
        if len(stripped) < self.min_transcript_chars:
            return False, ""
        return True, stripped

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

                should_process, trimmed = self._should_process(transcript)
                if not should_process:
                    self.logger.info("Ignored short/wakeword transcript: %r", transcript)
                    continue

                self.logger.info("Recognized: %s", trimmed)

                llm = self._get_llm_service()
                if llm is not None:
                    reply = llm.generate_simple(trimmed)
                    self.logger.info("LLM reply: %s", reply)
                    self._push_text(trimmed)

                    if hasattr(self.state, "history"):
                        self.state.history.append({"role": "user", "content": trimmed})
                        self.state.history.append({"role": "assistant", "content": reply})

            except Exception as exc:
                self.logger.exception("Voice loop processing failed: %s", exc)


__all__ = ["VoiceLoopService"]
