from __future__ import annotations

import difflib
import re
import time
from typing import Any, List, Optional

from core.logger import get_logger
from core.state import AppState
from services.base_service import BaseService
from utils.file_store import FileStore

class MemoryService(BaseService):
    name = "memory"

    def __init__(self, runtime, storage: Optional[FileStore] = None, file_name: str = "long_term_memory.json", limit: int = 100):
        super().__init__(runtime)
        self.state: AppState = runtime.state
        self.logger = get_logger("services.memory")
        self.storage = storage
        self.file_name = file_name
        self.limit = limit

    def start(self) -> None:
        self.state.long_term_memory = self.load()
        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        self.save(self.state.long_term_memory)
        self.state.set_service_status(self.name, "stopped")

    def load(self) -> List[dict]:
        if self.storage is None:
            return self.state.long_term_memory
        items = self.storage.load_json(self.file_name, default=[])
        if not isinstance(items, list):
            return []
        cleaned = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            cleaned.append({
                "content": content.strip(),
                "category": str(item.get("category", "general")),
                "importance": max(1, min(5, int(item.get("importance", 3)))),
                "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.7)))),
                "created_at": str(item.get("created_at", "")),
                "updated_at": str(item.get("updated_at", item.get("created_at", ""))),
                "use_count": max(0, int(item.get("use_count", 0))),
            })
        return cleaned[-self.limit:]

    def save(self, items: List[dict]) -> None:
        if self.storage is None:
            self.state.long_term_memory = items
            return
        self.storage.write_json(self.file_name, items[-self.limit:])

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[\W_]+", "", str(text).lower(), flags=re.UNICODE)

    @staticmethod
    def _is_sensitive(text: str) -> bool:
        patterns = [
            r"パスワード|password|passwd|シークレットキー|secret.?key|api.?key|アクセストークン|access.?token",
            r"クレジットカード|カード番号|口座番号|マイナンバー|秘密の質問",
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _memory_candidate(text: str) -> Optional[dict]:
        clean = re.sub(r"\s+", " ", str(text)).strip()
        if not clean or len(clean) < 5 or len(clean) > 500 or MemoryService._is_sensitive(clean):
            return None

        explicit_patterns = [
            r"覚えて(?:おいて(?:ね)?|てね|てください|ね)",
            r"(?:^|[、,:： ])覚えて(?:[、,:： ]|$)",
            r"記憶して", r"今後は", r"これからは", r"次回から", r"毎回", r"いつも", r"忘れないで",
        ]
        project_patterns = [
            r"作っている", r"開発している", r"制作している", r"作業中", r"プロジェクト",
            r"目的は", r"目標は", r"環境は", r"構成は", r"設定は", r"使用している",
            r"使っている", r"利用している", r"現在の状態", r"現在は", r"今は.*設定",
        ]
        preference_patterns = [
            r"が好き", r"が嫌い", r"好み", r"希望", r"してほしい", r"しないで", r"避けて",
            r"日本語で", r"短く", r"詳しく", r"口調", r"回答は", r"返答は",
        ]

        explicit = any(re.search(pattern, clean, re.IGNORECASE) for pattern in explicit_patterns)
        project = any(re.search(pattern, clean, re.IGNORECASE) for pattern in project_patterns)
        preference = any(re.search(pattern, clean, re.IGNORECASE) for pattern in preference_patterns)

        if not (explicit or project or preference):
            return None

        category = "preference" if preference else ("project" if project else "explicit")
        importance = 5 if explicit else (4 if project or preference else 3)
        confidence = 0.95 if explicit else (0.82 if project or preference else 0.7)
        return {"content": clean, "category": category, "importance": importance, "confidence": confidence}

    def remember_if_important(self, user_text: str) -> bool:
        candidate = self._memory_candidate(user_text)
        if not candidate:
            return False

        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        normalized = self._normalize(candidate["content"])
        for item in self.state.long_term_memory:
            existing = self._normalize(item.get("content", ""))
            if not existing:
                continue
            ratio = difflib.SequenceMatcher(None, normalized, existing).ratio()
            if normalized == existing or ratio >= 0.88 or normalized in existing or existing in normalized:
                if candidate["importance"] >= item.get("importance", 1):
                    item["content"] = candidate["content"]
                    item["category"] = candidate["category"]
                    item["importance"] = candidate["importance"]
                    item["confidence"] = max(item.get("confidence", 0.0), candidate["confidence"])
                    item["updated_at"] = now
                    self.save(self.state.long_term_memory)
                return False

        candidate.update({"created_at": now, "updated_at": now, "use_count": 0})
        self.state.long_term_memory.append(candidate)
        self.state.long_term_memory.sort(key=lambda item: (item.get("importance", 1), item.get("updated_at", "")), reverse=True)
        del self.state.long_term_memory[self.limit:]
        self.save(self.state.long_term_memory)
        return True

    def search(self, question: str, limit: int = 5) -> str:
        if not self.state.long_term_memory:
            return ""

        terms = set(re.findall(r"[一-龥ぁ-んァ-ヶー]{2,}|[a-zA-Z0-9]{3,}", question.lower()))
        if not terms:
            return ""

        scored = []
        for item in self.state.long_term_memory:
            content = item.get("content", "")
            searchable = content.lower()
            overlap = sum(1 for term in terms if term in searchable)
            if overlap:
                score = overlap + item.get("importance", 1) * 0.25 + item.get("confidence", 0.5) * 0.2
                scored.append((score, item))

        if not scored:
            return ""

        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = [item for _, item in scored[:limit]]

        for item in selected:
            item["use_count"] = item.get("use_count", 0) + 1

        return "長期記憶（参考情報）:\n" + "\n".join(
            f"- [{item.get('category', 'general')}] {item['content']}" for item in selected
        )