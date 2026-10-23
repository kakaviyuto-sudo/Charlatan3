from __future__ import annotations

from typing import Any, List, Optional, Dict

from openai import OpenAI

from core.logger import get_logger
from services.base_service import BaseService


class LLMService(BaseService):
    """
    Ollama-backed LLM client for the original Neuro-style assistant.

    This matches the original script setup:
      client = OpenAI(
          base_url="http://localhost:11434/v1",
          api_key="ollama",
          timeout=float(os.environ.get("NEURO_OLLAMA_TIMEOUT", "120")),
          max_retries=1,
      )
    """

    name = "llm"

    def __init__(
        self,
        runtime,
        model_name: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        timeout: float = 120.0,
        max_retries: int = 1,
        default_num_ctx: int = 4096,
    ):
        super().__init__(runtime)
        self.logger = get_logger("services.llm")
        self.model_name = model_name
        self.DEFAULT_NUM_CTX = max(2048, int(default_num_ctx))
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def start(self) -> None:
        self.logger.info("LLM service started")
        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        self.logger.info("LLM service stopped")
        self.state.set_service_status(self.name, "stopped")

    def generate_text(
        self,
        messages: List[dict],
        temperature: float = 0.55,
        max_tokens: int = 120,
        stream: bool = False,
        extra_body: Optional[dict] = None,
    ):
        """
        Calls the Ollama-compatible API with the same general settings as the original project.
        """
        options = {
            "options": {
                "num_ctx": self.DEFAULT_NUM_CTX,
                "num_predict": max_tokens,
                "repeat_penalty": 1.18,
            }
        }

        if extra_body:
            # Merge deep dicts conservatively
            if isinstance(extra_body, dict):
                for key, value in extra_body.items():
                    if key == "options" and isinstance(value, dict):
                        options["options"].update(value)
                    else:
                        options[key] = value

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            extra_body=options,
        )

        if stream:
            return response

        return response.choices[0].message.content or ""

    def generate_simple(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.generate_text(messages)

    def generate_chat(self, messages: List[dict], *, temperature: float = 0.55, max_tokens: int = 120) -> str:
        return self.generate_text(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )