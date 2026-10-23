from __future__ import annotations

from typing import Any, Dict

from llm.memory import MemoryService
from llm.service import LLMService
from services.service_registry import ServiceRegistry


def build_runtime_services(runtime, registry: ServiceRegistry) -> Dict[str, Any]:
    """
    Build the default runtime services for the live subsystem.
    Right now the app boots in minimal mode with a working memory service and an LLM service.
    """
    services: Dict[str, Any] = {}

    if "memory" not in registry._services:
        registry.register("memory", MemoryService)
    if "llm" not in registry._services:
        registry.register("llm", LLMService)

    services["memory"] = registry.create("memory", runtime=runtime)
    services["llm"] = registry.create("llm", runtime=runtime)

    if runtime.settings.enable_screen_monitor:
        runtime.logger.info("Screen monitor feature enabled; add screen service wiring here.")
    if runtime.settings.enable_youtube_live_mode:
        runtime.logger.info("YouTube live mode enabled; add YouTube service wiring here.")
    if runtime.settings.enable_web_search:
        runtime.logger.info("Web search enabled; add browser/service wiring here.")
    if runtime.settings.enable_vtube:
        runtime.logger.info("VTube feature enabled; add VTube service wiring here.")

    return services
