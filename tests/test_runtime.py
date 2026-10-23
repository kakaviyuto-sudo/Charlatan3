from config.settings import Settings
from app.runtime import Runtime


def test_runtime_can_start_in_minimal_mode():
    settings = Settings(
        enable_vtube=False,
        enable_screen_monitor=False,
        enable_youtube_live_mode=False,
        enable_web_search=False,
    )
    runtime = Runtime(settings)
    runtime.start()
    runtime.stop()
    assert runtime.state.shutdown_event.is_set()


def test_llm_service_rejects_empty_prompt():
    service = __import__("llm.service", fromlist=["LLMService"]).LLMService(runtime=None)
    assert service.generate("   ") == ""
