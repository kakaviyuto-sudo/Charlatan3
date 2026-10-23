from app.runtime import Runtime
from config.env import load_settings
from services.audio_service import AudioService
from services.recognition_service import RecognitionService
from services.stt_loader import load_stt_client

settings = load_settings()
runtime = Runtime(settings)

audio_service = AudioService(runtime, sample_rate=settings.sample_rate)
runtime.register_service("audio", audio_service)

stt_client = load_stt_client()
recognition_service = RecognitionService(runtime, stt_client=stt_client)
runtime.register_service("recognition", recognition_service)

runtime.start()
try:
    runtime.run_forever()
finally:
    runtime.stop()