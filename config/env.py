import os

from config.settings import Settings

def load_settings() -> Settings:
    return Settings(
        ollama_model=os.environ.get("NEURO_OLLAMA_MODEL", "qwen2.5:3b"),
        vision_model=os.environ.get("NEURO_VISION_MODEL", "qwen2.5vl:3b"),
        sample_rate=int(os.environ.get("NEURO_SAMPLE_RATE", "16000")),
        history_limit=int(os.environ.get("NEURO_HISTORY_LIMIT", "200")),
        sbv2_url=os.environ.get("SBV2_URL", "http://127.0.0.1:5000").rstrip("/"),
        sbv2_model_id=int(os.environ.get("SBV2_MODEL_ID", "0")),
        sbv2_speaker_id=int(os.environ.get("SBV2_SPEAKER_ID", "0")),
        sbv2_style=os.environ.get("SBV2_STYLE", "Neutral"),
        sbv2_language=os.environ.get("SBV2_LANGUAGE", "JP"),
    )