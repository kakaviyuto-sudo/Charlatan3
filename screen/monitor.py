from __future__ import annotations
import threading
import time

from core.state import AppState
from core.logger import get_logger

class ScreenMonitor:
    def __init__(self, state: AppState):
        self.state = state
        self.logger = get_logger("screen.monitor")
        self.stop_event = threading.Event()
        self.thread = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._worker, daemon=True, name="ScreenMonitor")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            # Placeholder: screenshot, compare with previous frame, emit event if changed
            time.sleep(6.0)