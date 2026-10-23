from __future__ import annotations
import time

class YouTubeLiveSession:
    def __init__(self):
        self.video_id = ""
        self.url = ""
        self.title = ""
        self.pending = []
        self.transcript = []
        self.started_at = ""

    def start(self, url: str, title: str) -> None:
        self.url = url
        self.title = title
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def add_caption(self, text: str) -> None:
        self.pending.append(text)

    def flush_pending(self) -> str:
        if not self.pending:
            return ""
        result = " ".join(self.pending)
        self.pending.clear()
        return result