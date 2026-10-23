from __future__ import annotations

import io
import queue
import threading
import time
from typing import Optional

import numpy as np
import soundfile as sf
import requests

from audio.output import AudioOutput
from core.logger import get_logger
from core.state import AppState
from services.base_service import BaseService


class TTSService(BaseService):
    name = "tts"

    def __init__(
        self,
        runtime,
        base_url: str = "http://127.0.0.1:5000",
        model_id: int = 0,
        speaker_id: int = 0,
        style: str = "Neutral",
        language: str = "JP",
    ):
        super().__init__(runtime)
        self.state: AppState = runtime.state
        self.logger = get_logger("services.tts")
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.speaker_id = speaker_id
        self.style = style
        self.language = language
        self.session = requests.Session()
        self.output = AudioOutput()
        self.queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True, name="TTSWorker")
        self.worker.start()
        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self.worker:
            self.worker.join(timeout=5.0)
        self.state.set_service_status(self.name, "stopped")

    def enqueue(self, text: str) -> None:
        if not text or not text.strip():
            return
        try:
            self.queue.put_nowait(text.strip())
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(text.strip())
            except queue.Full:
                self.logger.warning("TTS queue full; dropped text")

    def synthesize(self, text: str, timeout: int = 120):
        params = {
            "text": text,
            "model_id": self.model_id,
            "speaker_id": self.speaker_id,
            "sdp_ratio": 0.2,
            "noise": 0.6,
            "noisew": 0.8,
            "length": 1.0,
            "language": self.language,
            "auto_split": "true",
            "split_interval": 0.5,
            "assist_text_weight": 1.0,
            "style": self.style,
            "style_weight": 1.0,
        }
        response = self.session.post(
            f"{self.base_url}/voice",
            params=params,
            headers={"accept": "audio/wav"},
            timeout=timeout,
        )
        response.raise_for_status()
        audio, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        if audio.size == 0:
            raise RuntimeError("SBV2 returned empty audio")
        return audio, sample_rate

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.3)
            except queue.Empty:
                continue

            if item is None:
                break

            text = str(item).strip()
            if not text:
                continue

            try:
                audio, sample_rate = self.synthesize(text)
                self.output.play(audio, sample_rate)
            except Exception as exc:
                self.logger.exception("TTS synthesis/play failed: %s", exc)