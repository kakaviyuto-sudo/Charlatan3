from __future__ import annotations

from collections import deque
import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from core.logger import get_logger
from core.state import AppState
from services.base_service import BaseService


class AudioService(BaseService):
    """
    Original behavior preserved:
      - continuously listen to microphone
      - detect voice activity
      - accumulate utterance audio
      - push utterance to audio_queue when silence threshold reached
    """

    name = "audio"

    def __init__(
        self,
        runtime,
        sample_rate: int = 16000,
        threshold: float = 0.010,
        silence_limit: float = 0.9,
        max_utterance_seconds: float = 15.0,
    ):
        super().__init__(runtime)
        self.state: AppState = runtime.state
        self.logger = get_logger("services.audio")
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_limit = silence_limit
        self.max_utterance_seconds = max_utterance_seconds

        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.audio_queue: Optional[queue.Queue] = None
        self.block_seconds = 0.1

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return

        self.audio_queue = queue.Queue(maxsize=30)
        self.state.audio_queue = self.audio_queue
        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._capture_worker,
            name="MicrophoneCaptureWorker",
            daemon=True,
        )
        self.thread.start()

        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
        self.state.set_service_status(self.name, "stopped")

    def _queue_audio_utterance(self, chunks: list[np.ndarray]) -> None:
        if not chunks:
            return

        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        if self.audio_queue is None:
            return

        try:
            self.audio_queue.put_nowait(audio)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(audio)
            except queue.Full:
                self.logger.warning("Audio queue full; dropped utterance")

    def _capture_worker(self) -> None:
        block_frames = int(self.sample_rate * self.block_seconds)
        pre_roll = deque(maxlen=4)

        chunks = []
        speaking = False
        silence_time = 0.0
        recorded_time = 0.0

        while not self.stop_event.is_set():
            try:
                self.logger.info("常時音声認識: マイク監視中です。")
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=block_frames,
                ) as stream:
                    while not self.stop_event.is_set():
                        data, overflowed = stream.read(block_frames)
                        chunk = np.squeeze(data).copy()
                        volume = float(np.sqrt(np.mean(chunk ** 2)))

                        if not speaking:
                            pre_roll.append(chunk)
                            if volume >= self.threshold:
                                speaking = True
                                chunks = list(pre_roll)
                                recorded_time = len(chunks) * self.block_seconds
                                silence_time = 0.0
                                self.logger.info("音声を検出しました。認識をバックグラウンドで実行します。")
                        else:
                            chunks.append(chunk)
                            recorded_time += self.block_seconds
                            if volume < self.threshold:
                                silence_time += self.block_seconds
                            else:
                                silence_time = 0.0

                            if (
                                silence_time >= self.silence_limit
                                or recorded_time >= self.max_utterance_seconds
                            ):
                                self._queue_audio_utterance(chunks)
                                chunks = []
                                speaking = False
                                silence_time = 0.0
                                recorded_time = 0.0
                                pre_roll.clear()

            except Exception as exc:
                if not self.stop_event.is_set():
                    self.logger.exception("マイク常時監視エラー: %s", exc)
                time.sleep(1.0)
