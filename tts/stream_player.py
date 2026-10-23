from __future__ import annotations
import queue
import threading
import time

from audio.output import AudioOutput
from tts.sbv2_client import SBV2Client

class StreamPlayer:
    def __init__(self, tts_client: SBV2Client):
        self.tts_client = tts_client
        self.audio_output = AudioOutput()
        self.queue = queue.Queue(maxsize=12)
        self._thread = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.queue.put(None)

    def enqueue(self, text: str) -> None:
        self.queue.put(text)

    def _worker(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                break
            audio, sample_rate = self.tts_client.synthesize(item)
            self.audio_output.play(audio, sample_rate)