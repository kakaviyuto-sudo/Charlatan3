from __future__ import annotations
import numpy as np
import sounddevice as sd

class AudioOutput:
    def __init__(self):
        pass

    def play(self, audio: np.ndarray, sample_rate: int, device: str | None = None) -> None:
        # Placeholder for existing play_voice_audio logic
        if device is not None:
            sd.play(audio, samplerate=sample_rate, device=device)
        else:
            sd.play(audio, samplerate=sample_rate)
        sd.wait()