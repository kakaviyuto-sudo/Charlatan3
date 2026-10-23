from __future__ import annotations
import queue
import threading
import time

from core.state import AppState
from core.logger import get_logger

class RecognitionWorker:
    def __init__(self, state: AppState, stt):
        self.state = state
        self.stt = stt
        self.logger = get_logger("audio.recognition")
        self.stop_event = threading.Event()
        self.thread = None
        self.queue = queue.Queue(maxsize=100)

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.state.recognized_text_queue = self.queue
        self.thread = threading.Thread(target=self._worker, daemon=True, name="SpeechRecognition")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                audio = self.state.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                # Placeholder: call stt.transcribe(...) here
                text = ""
                if text:
                    self.queue.put_nowait(text)
            except Exception as exc:
                self.logger.exception("Recognition failed: %s", exc)
            finally:
                self.state.audio_queue.task_done()