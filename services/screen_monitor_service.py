from __future__ import annotations

import base64
import io
import threading
import time
from typing import Optional

import mss
from PIL import Image, ImageChops, ImageStat

from core.logger import get_logger
from core.state import AppState
from services.base_service import BaseService

class ScreenMonitorService(BaseService):
    name = "screen_monitor"

    def __init__(self, runtime, check_interval: float = 6.0, api_url: str = "http://localhost:11434/api/chat"):
        super().__init__(runtime)
        self.state: AppState = runtime.state
        self.logger = get_logger("services.screen_monitor")
        self.check_interval = check_interval
        self.api_url = api_url
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.last_frame = None
        self.recent_comments = []

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="ScreenMonitor")
        self.thread.start()
        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
        self.state.set_service_status(self.name, "stopped")

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            if time.monotonic() - self.state.last_screen_check_at >= self.check_interval:
                self.state.last_screen_check_at = time.monotonic()
                try:
                    comment = self._capture_and_comment()
                    if comment:
                        self.logger.info("Screen comment: %s", comment)
                        self.runtime.emit("screen_comment", comment)
                except Exception as exc:
                    self.logger.exception("Screen monitor failed: %s", exc)
            time.sleep(0.2)

    def _capture_and_comment(self) -> str:
        """
        Placeholder implementation that checks the active monitor and determines if a meaningful screen change happened.
        Replace this with your real screenshot + VLM logic.
        """
        try:
            with mss.mss() as sct:
                if len(sct.monitors) < 2:
                    return ""

                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                image.thumbnail((768, 432))

                current_frame = image.convert("L").resize((160, 90))
                if self.last_frame is not None:
                    diff = ImageChops.difference(current_frame, self.last_frame)
                    mean_diff = ImageStat.Stat(diff).mean[0]
                    if mean_diff < 3.0:
                        return ""
                self.last_frame = current_frame.copy()

                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=65, optimize=True)
                image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as exc:
            self.logger.exception("Failed to capture screenshot: %s", exc)
            return ""

        # The actual visual model call belongs here.
        # You can call Ollama local API using the same logic from the original script.
        return ""