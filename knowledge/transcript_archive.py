from __future__ import annotations
import json
import re
import time

class TranscriptArchive:
    def __init__(self, storage):
        self.storage = storage
        self.cache = []

    def save_entry(self, text: str, source: str = "caption", mode: str = "normal") -> bool:
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": source,
            "mode": mode,
            "text": text.strip(),
        }
        self.cache.append(entry)
        return True

    def search(self, question: str, limit: int = 5) -> str:
        # Placeholder: keyword-based search against transcript archive
        return ""