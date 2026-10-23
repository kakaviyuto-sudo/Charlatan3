from __future__ import annotations

from typing import Any, Dict

from llm.memory import MemoryService
from services.service_registry import ServiceRegistry


def build_runtime_services(runtime, registry: ServiceRegistry) -> Dict[str, Any]:
    """
    Build the default runtime service set while respecting feature toggles.
    Optional subsystems are omitted unless explicitly wired in by the runtime config.
    """
    services: Dict[str, Any] = {}

    if "memory" not in registry._services:
        registry.register("memory", MemoryService)

    services["memory"] = registry.create("memory", runtime=runtime)

    # Minimal-mode runtime: keep only the memory service unless the app explicitly enables
    # optional external subsystems. These are intentionally not instantiated here.
    if runtime.settings.enable_screen_monitor:
        runtime.logger.info("Screen monitor feature enabled; add screen service wiring here.")
    if runtime.settings.enable_youtube_live_mode:
        runtime.logger.info("YouTube live mode enabled; add YouTube service wiring here.")
    if runtime.settings.enable_web_search:
        runtime.logger.info("Web search enabled; add browser/service wiring here.")
    if runtime.settings.enable_vtube:
        runtime.logger.info("VTube feature enabled; add VTube service wiring here.")

    return services
