from config.env import load_settings
from app.runtime import Runtime
from services.audio_service import AudioService
from services.llm_service import LLMService
from services.memory_service import MemoryService
from services.transcript_archive_service import TranscriptArchiveService
from utils.file_store import FileStore


def main() -> None:
    settings = load_settings()
    runtime = Runtime(settings)

    store = FileStore("knowledge")

    # Minimal runtime wiring required for the lifecycle to have actual services.
    runtime.register_service(
        "audio",
        AudioService(runtime, sample_rate=settings.sample_rate),
    )
    runtime.register_service(
        "memory",
        MemoryService(runtime, storage=store, file_name="long_term_memory.json", limit=100),
    )
    runtime.register_service(
        "transcript_archive",
        TranscriptArchiveService(runtime, file_path="knowledge/all_audio_transcripts.jsonl"),
    )
    runtime.register_service(
        "llm",
        LLMService(runtime, model_name=settings.ollama_model),
    )

    try:
        runtime.start()
        runtime.run_forever()
    except KeyboardInterrupt:
        print("Shutdown requested by user.")
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
