from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Settings:
    ollama_model: str = "qwen2.5:3b"
    vision_model: str = "qwen2.5vl:3b"
    sample_rate: int = 16000
    history_limit: int = 200
    idle_chat_interval: int = 60
    screen_check_interval: float = 6.0
    sbv2_url: str = "http://127.0.0.1:5000"
    sbv2_model_id: int = 0
    sbv2_speaker_id: int = 0
    sbv2_style: str = "Neutral"
    sbv2_language: str = "JP"
    vts_url: str = "ws://127.0.0.1:8001"
    vts_talk_start_hotkey: str = "AI_Talking"
    vts_talk_end_hotkey: str = "AI_Idle"
    jimaku_bridge_port: int = 8765
    enable_vtube: bool = True
    enable_screen_monitor: bool = True
    enable_youtube_live_mode: bool = True
    enable_web_search: bool = True

    def __post_init__(self) -> None:
        if not self.ollama_model.strip():
            raise ValueError("ollama_model cannot be empty")
        if not self.vision_model.strip():
            raise ValueError("vision_model cannot be empty")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.history_limit <= 0:
            raise ValueError("history_limit must be > 0")
        if self.idle_chat_interval <= 0:
            raise ValueError("idle_chat_interval must be > 0")
        if self.screen_check_interval <= 0:
            raise ValueError("screen_check_interval must be > 0")
        if not self.sbv2_url.startswith(("http://", "https://")):
            raise ValueError("sbv2_url must be a valid HTTP(S) URL")
        if not self.vts_url.startswith(("ws://", "wss://")):
            raise ValueError("vts_url must be a valid WebSocket URL")
        if self.jimaku_bridge_port <= 0 or self.jimaku_bridge_port > 65535:
            raise ValueError("jimaku_bridge_port must be between 1 and 65535")
        if not isinstance(self.enable_vtube, bool):
            raise TypeError("enable_vtube must be a bool")
        if not isinstance(self.enable_screen_monitor, bool):
            raise TypeError("enable_screen_monitor must be a bool")
        if not isinstance(self.enable_youtube_live_mode, bool):
            raise TypeError("enable_youtube_live_mode must be a bool")
        if not isinstance(self.enable_web_search, bool):
            raise TypeError("enable_web_search must be a bool")
