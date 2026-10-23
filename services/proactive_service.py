from __future__ import annotations
import requests

from core.state import AppState

class ProactiveService:
    def __init__(self, state: AppState):
        self.state = state

    def generate(self) -> str:
        # Placeholder: fetch trends, generate proactive message
        return ""

    def fetch_trends(self) -> str:
        response = requests.get("https://trends.google.com/trending/rss?geo=JP", timeout=15)
        response.raise_for_status()
        return response.text