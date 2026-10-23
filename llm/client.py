from __future__ import annotations
from openai import OpenAI

class LLMClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0, max_retries: int = 1):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def chat(self, messages, model: str, temperature: float = 0.55, max_tokens: int = 120, stream: bool = False, **kwargs):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

    def stream_chat(self, messages, model: str, **kwargs):
        return self.chat(messages, model=model, stream=True, **kwargs)