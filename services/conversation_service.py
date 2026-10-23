from __future__ import annotations
from core.state import AppState
from llm.conversation import ConversationEngine
from llm.memory import MemoryService
from knowledge.transcript_archive import TranscriptArchive

class ConversationService:
    def __init__(self, state: AppState, conversation_engine: ConversationEngine, memory: MemoryService, transcript_archive: TranscriptArchive):
        self.state = state
        self.conversation_engine = conversation_engine
        self.memory = memory
        self.transcript_archive = transcript_archive

    def handle_user_text(self, user_text: str) -> str:
        # 1. remember if important
        self.memory.remember_if_important(user_text)

        # 2. search memory
        memory_context = self.memory.search(user_text)

        # 3. search transcript archive
        transcript_context = self.transcript_archive.search(user_text)

        # 4. generate response
        reply = self.conversation_engine.generate_text(
            user_text=user_text,
            history=self.state.history,
            memory_context=memory_context,
            transcript_context=transcript_context,
            evidence=""
        )

        # 5. save assistant reply
        self.state.history.append({"role": "assistant", "content": reply})
        self.state.history = self.state.history[-self.state.settings.history_limit:]
        return reply