from __future__ import annotations
from typing import List, Dict

class YouTubeKnowledgeStore:
    def __init__(self, storage):
        self.storage = storage

    def load(self) -> List[dict]:
        return self.storage.load_json("youtube_knowledge.json", default=[])

    def save(self, items: List[dict]) -> None:
        self.storage.write_json("youtube_knowledge.json", items)

    def index(self, items: List[dict]) -> List[tuple]:
        return [(item, "searchable text") for item in items]

    def search(self, question: str, limit: int = 2) -> str:
        return ""