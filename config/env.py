from __future__ import annotations

import os

from config.settings import Settings


def load_settings() -> Settings:
    return Settings(
        ollama_model=os.environ.get("NEURO_OLLAMA_MODEL", "qwen2.5:3b"),
        vision_model=os.environ.get("NEURO_VISION_MODEL", "qwen2.5vl:3b"),
        sample_rate=int(os.environ.get("NEURO_SAMPLE_RATE", "16000")),
        history_limit=int(os.environ.get("NEURO_HISTORY_LIMIT", "200")),
        idle_chat_interval=int(os.environ.get("NEURO_IDLE_CHAT_INTERVAL", "60")),
        screen_check_interval=float(os.environ.get("NEURO_SCREEN_CHECK_INTERVAL", "6.0")),
        sbv2_url=os.environ.get("SBV2_URL", "http://127.0.0.1:5000").rstrip("/"),
        sbv2_model_id=int(os.environ.get("SBV2_MODEL_ID", "0")),
        sbv2_speaker_id=int(os.environ.get("SBV2_SPEAKER_ID", "0")),
        sbv2_style=os.environ.get("SBV2_STYLE", "Neutral"),
        sbv2_language=os.environ.get("SBV2_LANGUAGE", "JP"),
        vts_url=os.environ.get("VTS_URL", "ws://127.0.0.1:8001"),
        vts_talk_start_hotkey=os.environ.get("VTS_TALK_START_HOTKEY", "AI_Talking"),
        vts_talk_end_hotkey=os.environ.get("VTS_TALK_END_HOTKEY", "AI_Idle"),
        jimaku_bridge_port=int(os.environ.get("JIMAKU_BRIDGE_PORT", "8765")),
        enable_vtube=os.environ.get("ENABLE_VTUBE", "true").strip().lower() in {"1", "true", "yes", "on"},
        enable_screen_monitor=os.environ.get("ENABLE_SCREEN_MONITOR", "true").strip().lower() in {"1", "true", "yes", "on"},
        enable_youtube_live_mode=os.environ.get("ENABLE_YOUTUBE_LIVE_MODE", "true").strip().lower() in {"1", "true", "yes", "on"},
        enable_web_search=os.environ.get("ENABLE_WEB_SEARCH", "true").strip().lower() in {"1", "true", "yes", "on"},
    )
