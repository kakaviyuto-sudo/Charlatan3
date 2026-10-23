from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Any, Callable, Dict, List, Optional, Type

from config.settings import Settings

@dataclass
class AppState:
    settings: Settings

    # process lifecycle
    shutdown_event: Event = field(default_factory=Event)
    lifecycle_lock: Lock = field(default_factory=Lock)

    # chat / memory
    history: List[dict] = field(default_factory=list)
    long_term_memory: List[dict] = field(default_factory=list)

    # knowledge
    youtube_knowledge: List[dict] = field(default_factory=list)
    web_knowledge: List[dict] = field(default_factory=list)
    wiki_knowledge: List[dict] = field(default_factory=list)
    streamer_style: str = ""
    phrase_style: str = ""

    # transcript archive
    transcript_archive: deque = field(default_factory=lambda: deque(maxlen=5000))
    transcript_last_signature: str = ""
    transcript_last_at: float = 0.0
    recent_ai_voice_signatures: deque = field(default_factory=lambda: deque(maxlen=30))

    # live youtube
    youtube_live_mode: bool = False
    youtube_live_pending: List[str] = field(default_factory=list)
    youtube_live_transcript: List[str] = field(default_factory=list)
    youtube_live_session: Dict[str, Any] = field(default_factory=dict)

    # audio
    audio_queue: Optional[Any] = None
    recognized_text_queue: Optional[Any] = None

    # service runtime
    services: Dict[str, Any] = field(default_factory=dict)
    service_order: List[str] = field(default_factory=list)
    service_status: Dict[str, str] = field(default_factory=dict)

    # timestamps
    last_interaction_at: float = 0.0
    last_screen_check_at: float = 0.0

    # optional resources
    prompt_context: Dict[str, Any] = field(default_factory=dict)

    def register_service(self, name: str, instance: Any) -> None:
        self.services[name] = instance
        if name not in self.service_order:
            self.service_order.append(name)

    def get_service(self, name: str) -> Any:
        return self.services.get(name)

    def set_service_status(self, name: str, status: str) -> None:
        self.service_status[name] = status

    def mark_interaction(self) -> None:
        import time
        self.last_interaction_at = time.monotonic()

    def mark_screen_check(self) -> None:
        import time
        self.last_screen_check_at = time.monotonic()

    def reset_runtime(self) -> None:
        self.shutdown_event.clear()
        self.history.clear()
        self.long_term_memory.clear()
        self.youtube_knowledge.clear()
        self.web_knowledge.clear()
        self.wiki_knowledge.clear()
        self.youtube_live_pending.clear()
        self.youtube_live_transcript.clear()
        self.youtube_live_session = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "AppState":
        return cls(settings=settings)