from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional


class MemoryService:
    def __init__(self, storage=None, runtime=None):
        self.storage = storage
        self.runtime = runtime
        self._items: List[Dict[str, Any]] = []
        self._load_from_storage()

    def start(self) -> None:
        self._load_from_storage()

    def stop(self) -> None:
        self.save(self._items)

    def tick(self) -> None:
        pass

    def _load_from_storage(self) -> None:
        if self.storage is None or not hasattr(self.storage, "load_json"):
            return
        try:
            items = self.storage.load_json("long_term_memory.json", default=[])
        except Exception:
            items = []
        if isinstance(items, list):
            self._items = items

    def remember_if_important(self, user_text: str) -> bool:
        text = (user_text or "").strip()
        if not text:
            return False

        normalized = re.sub(r"\s+", " ", text)
        if len(normalized) < 8:
            return False

        tokens = re.findall(r"\b\w+\b", normalized)
        if len(tokens) < 3:
            return False

        score = len(normalized) / 4 + len(tokens) * 2
        if "?" in normalized or "!" in normalized:
            score += 3

        if score < 14:
            return False

        entry = {
            "text": normalized,
            "importance": score,
            "timestamp": time.time(),
        }
        self._items.append(entry)
        self.save(self._items)
        return True

    def search(self, question: str, limit: int = 5) -> str:
        if not question:
            return ""

        query_tokens = {
            token.lower()
            for token in re.findall(r"\b\w+\b", question)
            if len(token) > 2
        }
        if not query_tokens:
            return ""

        scored: List[tuple[int, str]] = []
        for item in reversed(self._items):
            text = str(item.get("text", "")).lower()
            matches = sum(1 for token in query_tokens if token in text)
            if matches:
                scored.append((matches, text))

        if not scored:
            return ""

        scored.sort(key=lambda entry: entry[0], reverse=True)
        return "\n".join(text for _, text in scored[:limit])

    def load(self):
        self._load_from_storage()
        return list(self._items)

    def save(self, items):
        self._items = list(items or [])
        if self.storage is not None and hasattr(self.storage, "write_json"):
            self.storage.write_json("long_term_memory.json", self._items)
