from config.env import load_settings
from app.runtime import Runtime
from services.llm_service import LLMService

settings = load_settings()
runtime = Runtime(settings)

llm_service = LLMService(
    runtime=runtime,
    model_name=settings.ollama_model,
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=120.0,
    max_retries=1,
    default_num_ctx=4096,
)

runtime.register_service("llm", llm_service)