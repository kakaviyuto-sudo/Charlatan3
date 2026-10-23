from __future__ import annotations
import io
import soundfile as sf
import requests

class SBV2Client:
    def __init__(self, base_url: str, model_id: int, speaker_id: int, style: str, language: str):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.speaker_id = speaker_id
        self.style = style
        self.language = language
        self.session = requests.Session()

    def synthesize(self, text: str, timeout: int = 120):
        params = {
            "text": text,
            "model_id": self.model_id,
            "speaker_id": self.speaker_id,
            "language": self.language,
            "style": self.style,
            "auto_split": "true",
        }
        response = self.session.post(
            f"{self.base_url}/voice",
            params=params,
            headers={"accept": "audio/wav"},
            timeout=timeout,
        )
        response.raise_for_status()
        audio, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        return audio, sample_rate