from __future__ import annotations

import re
import time
from typing import List, Optional

from core.logger import get_logger
from core.state import AppState
from services.base_service import BaseService

class YouTubeLiveService(BaseService):
    name = "youtube_live"

    def __init__(self, runtime):
        super().__init__(runtime)
        self.state: AppState = runtime.state
        self.logger = get_logger("services.youtube_live")
        self.session = {
            "video_id": "",
            "url": "",
            "title": "",
            "started_at": "",
        }
        self.pending: List[str] = []
        self.transcript: List[str] = []
        self.last_caption_at = 0.0
        self.last_learned_length = 0

    def start(self) -> None:
        self.state.youtube_live_mode = False
        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        self.state.youtube_live_mode = False
        self.state.set_service_status(self.name, "stopped")

    def start_session(self, url: str, title: str) -> None:
        self.session.update({
            "video_id": re.search(r"v=([A-Za-z0-9_-]{11})", url).group(1) if "v=" in url else "",
            "url": url,
            "title": title,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self.pending.clear()
        self.transcript.clear()
        self.state.youtube_live_mode = True
        self.state.youtube_live_session = self.session.copy()
        self.state.youtube_live_pending = self.pending
        self.state.youtube_live_transcript = self.transcript
        self.logger.info("YouTube live session started: %s", url)

    def add_caption(self, text: str) -> None:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return

        if self.pending and self.pending[-1] == text:
            return
        if self.transcript and self.transcript[-1] == text:
            return

        self.pending.append(text)
        self.transcript.append(text)
        self.last_caption_at = time.monotonic()

        self.state.youtube_live_pending = self.pending
        self.state.youtube_live_transcript = self.transcript

    def flush_pending(self, force: bool = False) -> str:
        if not self.pending:
            return ""

        joined = " ".join(self.pending).strip()
        age = time.monotonic() - self.last_caption_at

        if not force and len(joined) < 55 and age < 4.0:
            return ""
        if not force and len(joined) < 18 and age < 8.0:
            return ""

        self.pending.clear()
        self.state.youtube_live_pending = self.pending
        return joined[:1800]

    def finish_session(self) -> None:
        if not self.state.youtube_live_mode:
            return

        batch = self.flush_pending(force=True)
        if batch:
            self.logger.info("Final YouTube live batch: %s", batch[:500])

        self.state.youtube_live_mode = False
        self.state.youtube_live_session = {}