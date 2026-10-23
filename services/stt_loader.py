from __future__ import annotations

def load_stt_client():
    try:
        from faster_whisper import WhisperModel
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as exc:
        return None