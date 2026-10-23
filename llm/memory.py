from __future__ import annotations
import difflib
import re
import time

class MemoryService:
    def __init__(self, storage):
        self.storage = storage

    def remember_if_important(self, user_text: str) -> bool:
        # Placeholder: implement candidate logic and save to long_term_memory
        return False

    def search(self, question: str, limit: int = 5) -> str:
        # Placeholder: keyword-overlap search against long-term memory
        return ""

    def load(self):
        return self.storage.load_json("long_term_memory.json", default=[])

    def save(self, items):
        self.storage.write_json("long_term_memory.json", items)