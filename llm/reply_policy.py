from __future__ import annotations
import re
import difflib

class ReplyPolicy:
    def __init__(self):
        pass

    @staticmethod
    def normalize(value: str) -> str:
        value = str(value or "").casefold()
        value = re.sub(r"https?://\\S+", " ", value)
        return re.sub(r"[\\s\\W_]+", "", value, flags=re.UNICODE)

    @classmethod
    def similarity(cls, left: str, right: str) -> float:
        a = cls.normalize(left)
        b = cls.normalize(right)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

    @classmethod
    def is_repetitive(cls, candidate: str, previous_replies: list[str], threshold: float = 0.8) -> bool:
        if not candidate.strip():
            return False
        return any(cls.similarity(candidate, prev) >= threshold for prev in previous_replies)