from __future__ import annotations
import queue
import threading
import time
import numpy as np
import sounddevice as sd

from core.state import AppState
from core.logger import get_logger

class MicrophoneListener:
    def __init__(self, state: AppState):
        self.state = state
        self.logger = get_logger("audio.microphone")
        self.stop_event = threading.Event()
        self.thread = None
        self.audio_queue = queue.Queue(maxsize=30)

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.state.audio_queue = self.audio_queue
        self.thread = threading.Thread(target=self._worker, daemon=True, name="MicrophoneListener")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)

    def _worker(self) -> None:
        # Placeholder: implement block-based microphone capture
        while not self.stop_event.is_set():
            time.sleep(0.1)