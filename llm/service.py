from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Optional

from services.base_service import BaseService


class LLMService(BaseService):
    name = "llm"

    def __init__(self, runtime=None, model: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 30.0):
        super().__init__(runtime=runtime)
        base = base_url or getattr(runtime.settings, "ollama_base_url", "http://127.0.0.1:11434") if runtime else "http://127.0.0.1:11434"
        self.model = model or getattr(runtime.settings, "ollama_model", "qwen2.5:3b") if runtime else "qwen2.5:3b"
        self.base_url = base.rstrip("/")
        self.timeout = timeout
        self._ready = False

    def start(self) -> None:
        self._ready = True

    def stop(self) -> None:
        self._ready = False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 512) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return ""

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if system_prompt:
            payload["system"] = system_prompt

        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception:
            return ""

        if isinstance(result, dict):
            if "response" in result:
                return str(result["response"]).strip()
            if "message" in result and isinstance(result["message"], dict):
                return str(result["message"].get("content", "")).strip()
        return ""

    def chat(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 512) -> str:
        return self.generate(prompt, system_prompt=system_prompt, max_tokens=max_tokens)
