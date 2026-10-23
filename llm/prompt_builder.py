from __future__ import annotations
from typing import List, Dict

class PromptBuilder:
    def __init__(self, system_prompt: str, stream_delivery_prompt: str):
        self.system_prompt = system_prompt
        self.stream_delivery_prompt = stream_delivery_prompt

    def build(self, user_text: str, style_instruction: str, memory_context: str = "", transcript_context: str = "", evidence: str = "", history: List[dict] = None) -> List[dict]:
        messages = [
            {"role": "system", "content": self.system_prompt + self.stream_delivery_prompt + style_instruction},
        ]
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        if transcript_context:
            messages.append({"role": "system", "content": transcript_context})
        if evidence:
            messages.append({"role": "system", "content": evidence})
        if history:
            messages.extend(history[-12:])
        messages.append({"role": "user", "content": user_text})
        return messages