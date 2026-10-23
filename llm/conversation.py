from __future__ import annotations
from typing import List

from llm.client import LLMClient
from llm.prompt_builder import PromptBuilder
from llm.reply_policy import ReplyPolicy

class ConversationEngine:
    def __init__(self, llm: LLMClient, prompt_builder: PromptBuilder):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.reply_policy = ReplyPolicy()

    def generate_text(self, user_text: str, history: List[dict], style_instruction: str = "", memory_context: str = "", transcript_context: str = "", evidence: str = "") -> str:
        messages = self.prompt_builder.build(
            user_text=user_text,
            style_instruction=style_instruction,
            memory_context=memory_context,
            transcript_context=transcript_context,
            evidence=evidence,
            history=history,
        )
        response = self.llm.chat(
            model="qwen2.5:3b",
            messages=messages,
            temperature=0.55,
            max_tokens=120,
        )
        text = response.choices[0].message.content or ""
        return text.strip()