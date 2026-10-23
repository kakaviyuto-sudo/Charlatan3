from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

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