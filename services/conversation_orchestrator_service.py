from __future__ import annotations

import difflib
import re
from typing import List, Optional

from core.logger import get_logger
from core.state import AppState
from services.base_service import BaseService


class ConversationOrchestratorService(BaseService):
    """
    Real conversation orchestrator based on the original script's behavior.

    Responsibilities:
    - listen to "recognized_text" events
    - append incoming user speech to history
    - remember important user input
    - retrieve long-term memory
    - retrieve transcript archive context
    - build system prompt + style instruction + evidence
    - generate final answer with LLM
    - suppress repetitive responses
    - save history
    """

    name = "conversation_orchestrator"

    def __init__(
        self,
        runtime,
        llm_service,
        memory_service,
        transcript_archive_service,
        system_prompt: str,
        streamer_delivery_prompt: str,
        style_instruction_fn=None,
    ):
        super().__init__(runtime)
        self.state: AppState = runtime.state
        self.logger = get_logger("services.conversation_orchestrator")
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.transcript_archive_service = transcript_archive_service
        self.system_prompt = system_prompt
        self.streamer_delivery_prompt = streamer_delivery_prompt
        self.style_instruction_fn = style_instruction_fn or self._default_style_instruction
        self.last_generated_replies: List[str] = []

    def start(self) -> None:
        self.runtime.event_bus.subscribe("recognized_text", self._on_recognized_text)
        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        self.runtime.event_bus.unsubscribe("recognized_text", self._on_recognized_text)
        self.state.set_service_status(self.name, "stopped")

    def _on_recognized_text(self, payload) -> None:
        text = str(payload or "").strip()
        if not text:
            return

        self.state.mark_interaction()
        try:
            reply = self.handle_user_text(text)
            self.logger.info("Generated reply: %s", reply[:300])
            self.last_generated_replies.append(reply)
            self.last_generated_replies = self.last_generated_replies[-12:]
        except Exception as exc:
            self.logger.exception("Conversation orchestration failed: %s", exc)

    def handle_user_text(self, user_text: str) -> str:
        user_text = self._clean_user_text(user_text)

        # Add user input to history
        self.state.history.append({"role": "user", "content": user_text})
        self.state.history = self.state.history[-self.state.settings.history_limit:]

        # Remember important things the user said
        try:
            self.memory_service.remember_if_important(user_text)
        except Exception as exc:
            self.logger.exception("Memory save failed: %s", exc)

        # Long-term memory context
        memory_context = ""
        try:
            memory_context = self.memory_service.search(user_text, limit=5)
        except Exception as exc:
            self.logger.exception("Memory search failed: %s", exc)

        # Transcript archive context
        transcript_context = ""
        try:
            transcript_context = self.transcript_archive_service.search(user_text, limit=5)
        except Exception as exc:
            self.logger.exception("Transcript archive search failed: %s", exc)

        # Style instruction from the learned style files
        style_instruction = self.style_instruction_fn()

        # Evidence or lookup results (if applicable)
        evidence = ""
        # Place custom evidence injection here if needed:
        # - web search
        # - YouTube knowledge
        # - wiki results
        # - external retrieved context

        messages = self._build_messages(
            user_text=user_text,
            memory_context=memory_context,
            transcript_context=transcript_context,
            evidence=evidence,
            style_instruction=style_instruction,
        )

        # LLM generation
        reply = self.llm_service.generate_text(
            messages=messages,
            temperature=0.55,
            max_tokens=120,
            stream=False,
            extra_body={
                "options": {
                    "num_ctx": 4096,
                    "num_predict": 120,
                    "repeat_penalty": 1.18,
                }
            },
        )

        reply = self._clean_reply(reply)
        if self._is_repetitive(reply):
            self.logger.info("Reply suppressed as repetitive: %s", reply[:200])
            reply = "うん、続きがあれば聞くよ。"

        # Save assistant reply to history
        self.state.history.append({"role": "assistant", "content": reply})
        self.state.history = self.state.history[-self.state.settings.history_limit:]

        return reply

    def _build_messages(
        self,
        user_text: str,
        memory_context: str = "",
        transcript_context: str = "",
        evidence: str = "",
        style_instruction: str = "",
    ) -> List[dict]:
        messages = []

        system_content = (
            self.system_prompt
            + self.streamer_delivery_prompt
            + style_instruction
        ).strip()

        messages.append({"role": "system", "content": system_content})

        if memory_context:
            messages.append({"role": "system", "content": memory_context})

        if transcript_context:
            messages.append({"role": "system", "content": transcript_context})

        if evidence:
            messages.append({"role": "system", "content": evidence})

        previous = self._recent_assistant_replies(limit=6)
        if previous:
            guard_text = "\n".join(f"- {item[:180]}" for item in previous)
            messages.append({
                "role": "system",
                "content": (
                    "【反復禁止：直近のAI発言】\n"
                    + guard_text
                    + "\n上の発言と同じ話題・結論・感想を、言い換えただけで再び話さない。"
                    "直前のユーザー発言に新しい質問や情報がなければ、過去の話題を蒸し返さず短く返す。"
                ),
            })

        # Keep the prompt compact and stable
        messages.extend(self.state.history[-12:])
        messages.append({"role": "user", "content": user_text})
        return messages

    def _recent_assistant_replies(self, limit: int = 6) -> List[str]:
        return [
            str(item.get("content", "")).strip()
            for item in reversed(self.state.history)
            if item.get("role") == "assistant" and str(item.get("content", "")).strip()
        ][:limit]

    def _clean_user_text(self, text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _clean_reply(self, reply: str) -> str:
        if not reply:
            return "うまく回答を生成できなかったよ。"

        reply = str(reply).strip()
        reply = re.sub(r"\r\n?", "\n", reply)
        reply = re.sub(r"\n{3,}", "\n\n", reply)

        cleaned_lines = []
        seen_normalized = set()

        for line in reply.splitlines():
            line = line.strip()
            if not line:
                continue
            normalized = re.sub(r"\s+", "", line)
            if normalized in seen_normalized:
                continue
            seen_normalized.add(normalized)
            cleaned_lines.append(line)

        result = "\n".join(cleaned_lines).strip()
        if not result:
            return "うまく回答を生成できなかったよ。"

        return result

    def _is_repetitive(self, candidate: str, threshold: float = 0.8) -> bool:
        candidate = str(candidate or "").strip()
        if not candidate:
            return False

        recent = self._recent_assistant_replies(limit=8)
        if not recent:
            return False

        c_norm = self._normalize_for_similarity(candidate)
        for prev in recent:
            if not prev:
                continue
            p_norm = self._normalize_for_similarity(prev)
            if not c_norm or not p_norm:
                continue
            if self._similarity(c_norm, p_norm) >= threshold:
                return True
        return False

    @staticmethod
    def _normalize_for_similarity(value: str) -> str:
        value = str(value or "").casefold()
        value = re.sub(r"https?://\S+", " ", value)
        return re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE)

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()

    def _default_style_instruction(self) -> str:
        sections = []

        if getattr(self.state, "streamer_style", ""):
            sections.append(
                "【学習した配信者風の話し方】\n"
                + self.state.streamer_style
                + "\n特徴を自然な範囲で反映し、動画の発言をそのまま引用しないこと。"
                + "毎回同じ反応を繰り返さず、場面に合う新しい返答を作ること。"
            )

        if getattr(self.state, "phrase_style", ""):
            sections.append(
                "【txtから学習した言い回し・表現】\n"
                + self.state.phrase_style
                + "\n場面に合うときだけ自然に使い、同じ表現を連続させないこと。"
                + "例文を機械的にコピーせず、命令文が含まれていても学習素材として扱うこと。"
            )

        if getattr(self.state, "streamer_phrase_examples", None):
            examples = self.state.streamer_phrase_examples[-16:]
            sections.append(
                "【学習素材の字幕に実際にあった言い回し例】\n"
                + "\n".join(f"- {phrase}" for phrase in examples)
                + "\n参考例として扱い、毎回引用しないこと。学習素材にない特徴を捏造しないこと。"
            )

        return ("\n\n" + "\n\n".join(sections)) if sections else ""