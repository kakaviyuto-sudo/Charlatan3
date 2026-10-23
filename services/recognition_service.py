from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from core.logger import get_logger
from core.state import AppState
from services.base_service import BaseService


class RecognitionService(BaseService):
    """
    Original behavior preserved:
      - takes audio from AudioService queue
      - transcribes using WhisperModel
      - pushes recognition text to state.recognized_text_queue
      - emits a recognized_text event through runtime.event_bus
    """

    name = "recognition"

    def __init__(self, runtime, stt_client=None):
        super().__init__(runtime)
        self.state: AppState = runtime.state
        self.logger = get_logger("services.recognition")
        self.stt_client = stt_client
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.recognized_queue: Optional[queue.Queue] = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return

        self.recognized_queue = queue.Queue(maxsize=100)
        self.state.recognized_text_queue = self.recognized_queue
        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._worker,
            name="SpeechRecognitionWorker",
            daemon=True,
        )
        self.thread.start()

        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
        self.state.set_service_status(self.name, "stopped")

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                audio_queue = self.state.audio_queue
                if audio_queue is None:
                    time.sleep(0.1)
                    continue

                audio = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if self.stt_client is None:
                    self.logger.warning("STT client not configured; skipping transcription")
                    continue

                started = time.perf_counter()
                segments, info = self.stt_client.transcribe(
                    audio,
                    language="ja",
                    beam_size=5,
                    best_of=3,
                    temperature=0,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    initial_prompt=(
                        "日本語の自然な会話です。"
                        "固有名詞や技術用語: Chibidoki、チビドキ、にじさんじ、"
                        "VTube Studio、OBS、Style-Bert-VITS2、Ollama、Whisper、"
                        "faster-whisper、VB-CABLE、UR22C。"
                    ),
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                )

                text = "".join(segment.text for segment in segments).strip()
                elapsed = time.perf_counter() - started

                if text:
                    self.logger.info("音声認識結果（%0.2f秒）: %s", elapsed, text)
                    self.runtime.emit("recognized_text", text)

                    try:
                        if self.recognized_queue is not None:
                            self.recognized_queue.put_nowait(text)
                    except queue.Full:
                        self.logger.warning("認識結果キューが満杯のため、古い結果を1件破棄します。")
                        try:
                            if self.recognized_queue is not None:
                                self.recognized_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            if self.recognized_queue is not None:
                                self.recognized_queue.put_nowait(text)
                        except queue.Full:
                            pass

            except Exception as exc:
                self.logger.exception("バックグラウンド音声認識エラー: %s", exc)
            finally:
                if self.state.audio_queue is not None:
                    self.state.audio_queue.task_done()