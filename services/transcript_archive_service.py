from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

from core.logger import get_logger
from core.state import AppState
from services.base_service import BaseService

class TranscriptArchiveService(BaseService):
    name = "transcript_archive"

    def __init__(self, runtime, file_path: str = "all_audio_transcripts.jsonl"):
        super().__init__(runtime)
        self.state: AppState = runtime.state
        self.logger = get_logger("services.transcript_archive")
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: deque = deque(maxlen=5000)

    def start(self) -> None:
        self.state.transcript_archive = self.cache
        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        self.state.set_service_status(self.name, "stopped")

    def save_entry(self, text: str, source: str = "caption", mode: str = "normal") -> bool:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return False

        now = time.time()
        signature = re.sub(r"[\s、。,.!?！？]+", "", text).casefold()

        # prevent duplicate caption entries within short period
        if source == "caption":
            if signature == self.state.transcript_last_signature and now - self.state.transcript_last_at < 8:
                return False
            if any(
                spoken_signature == signature and now - spoken_at <= 45
                for spoken_at, spoken_signature in self.state.recent_ai_voice_signatures
            ):
                return False

        if source == "ai_voice":
            self.state.recent_ai_voice_signatures.append((now, signature))

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "epoch": now,
            "source": source,
            "mode": mode,
            "text": text,
        }

        try:
            with self.file_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.cache.append(entry)
            if source == "caption":
                self.state.transcript_last_signature = signature
                self.state.transcript_last_at = now
            return True
        except OSError as exc:
            self.logger.exception("Could not write transcript entry: %s", exc)
            return False

    def search(self, question: str, limit: int = 5) -> str:
        terms = set(re.findall(r"[一-龥ぁ-んァ-ヶー]{2,}|[a-zA-Z0-9]{3,}", str(question).lower()))
        if not terms:
            return ""

        scored = []
        for item in list(self.cache)[-5000:]:
            if item.get("source") != "caption":
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            score = sum(1 for term in terms if term in text.lower())
            if score:
                scored.append((score, item))

        if not scored:
            return ""

        scored.sort(key=lambda pair: (pair[0], pair[1].get("epoch", 0)), reverse=True)
        selected = []
        seen = set()
        for _, item in scored:
            norm = re.sub(r"\s+", "", item.get("text", ""))
            if not norm or norm in seen:
                continue
            seen.add(norm)
            selected.append(item)
            if len(selected) >= limit:
                break

        if not selected:
            return ""

        return "過去の音声字幕アーカイブ（参考情報）:\n" + "\n".join(
            f"- [{x.get('timestamp', '時刻不明')} / {x.get('source', 'caption')}] {x.get('text', '')[:500]}"
            for x in selected
        )