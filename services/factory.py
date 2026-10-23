from __future__ import annotations

from typing import Any, Dict

from app.runtime import Runtime
from config.settings import Settings
from llm.memory import MemoryService
from services.service_registry import ServiceRegistry


def build_runtime_services(runtime: Runtime, registry: ServiceRegistry) -> Dict[str, Any]:
    """
    Wire the services required for the runtime to function with a small but real default set.
    """
    services: Dict[str, Any] = {}

    if "memory" not in registry._services:
        registry.register("memory", MemoryService)

    services["memory"] = registry.create("memory", runtime=runtime)
    return services
