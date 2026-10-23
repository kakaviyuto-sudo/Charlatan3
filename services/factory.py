from __future__ import annotations

from typing import Dict, Any

from app.runtime import Runtime
from config.settings import Settings
from services.service_registry import ServiceRegistry

def build_runtime_services(runtime: Runtime, registry: ServiceRegistry) -> Dict[str, Any]:
    """
    This function wires the actual services into the runtime.
    The concrete service classes will be added here when you implement each domain.
    """
    services: Dict[str, Any] = {}

    # Example placeholders:
    # services["llm"] = registry.create("llm", runtime)
    # services["memory"] = registry.create("memory", runtime)
    # services["audio"] = registry.create("audio", runtime)
    # services["tts"] = registry.create("tts", runtime)
    # services["vtube"] = registry.create("vtube", runtime)
    # services["screen"] = registry.create("screen", runtime)
    # services["bridge"] = registry.create("bridge", runtime)

    return services