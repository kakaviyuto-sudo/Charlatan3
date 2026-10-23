from __future__ import annotations

import io
import threading
import time
import wave
from typing import Optional

import numpy as np
from openai import OpenAI

from core.logger import get_logger
from services.base_service import BaseService


class STTService(BaseService):
    name = "stt"

    def __init__(
        self,
        runtime,
        model: str = "whisper-1",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        super().__init__(runtime)
        self.logger = get_logger("services.stt")
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker, name="STTWorker", daemon=True)
        self.thread.start()
        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
        self.state.set_service_status(self.name, "stopped")

    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio_i16 = np.clip(audio, -1.0, 1.0) * 32767
        audio_i16 = audio_i16.astype("<i2")

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_i16.tobytes())
        return wav_buffer.getvalue()

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            q = self.state.audio_queue
            if q is None:
                time.sleep(0.2)
                continue

            try:
                audio = q.get(timeout=0.5)
            except Exception:
                continue

            try:
                wav = self._to_wav_bytes(audio)
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=("utterance.wav", wav, "audio/wav"),
                )
                text = getattr(transcript, "text", "") or str(transcript)
                self.logger.info("Recognized: %s", text)

                if self.state.recognized_text_queue is None:
                    self.state.recognized_text_queue = []

                if hasattr(self.state.recognized_text_queue, "append"):
                    self.state.recognized_text_queue.append(text)
                else:
                    self.state.recognized_text_queue.put(text)

            except Exception as exc:
                self.logger.exception("STT failed: %s", exc)
