from config.env import load_settings
from app.runtime import Runtime
from services.audio_service import AudioService
from services.recognition_service import RecognitionService
from services.llm_service import LLMService
from services.memory_service import MemoryService
from services.transcript_archive_service import TranscriptArchiveService
from utils.file_store import FileStore

settings = load_settings()
runtime = Runtime(settings)

# storage
store = FileStore("knowledge")

# services
runtime.register_service("audio", AudioService(runtime, sample_rate=settings.sample_rate))
runtime.register_service("memory", MemoryService(runtime, storage=store, file_name="long_term_memory.json", limit=100))
runtime.register_service("transcript_archive", TranscriptArchiveService(runtime, file_path="knowledge/all_audio_transcripts.jsonl"))
runtime.register_service("llm", LLMService(runtime, client=None, model_name=settings.ollama_model))

# recognition requires an actual STT client object
# runtime.register_service("recognition", RecognitionService(runtime, stt_client=stt_client))

runtime.start()
try:
    runtime.run_forever()
finally:
    runtime.stop()