from config.env import load_settings
from app.runtime import Runtime
from services.service_registry import ServiceRegistry
from services.factory import build_runtime_services

settings = load_settings()
runtime = Runtime(settings)

registry = ServiceRegistry()
# register services later
# registry.register("llm", LLMService)
# registry.register("memory", MemoryService)
# registry.register("audio", AudioService)
# registry.register("tts", TTSService)
# registry.register("vtube", VTubeControllerService)

services = build_runtime_services(runtime, registry)
runtime.register_services(services)

runtime.start()
try:
    runtime.run_forever()
finally:
    runtime.stop()