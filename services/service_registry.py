from __future__ import annotations

from typing import Dict, Type

from services.base_service import BaseService


class ServiceRegistry:
    def __init__(self):
        self._services: Dict[str, Type[BaseService]] = {}

    def register(self, name: str, service_cls: Type[BaseService]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Service name must be a non-empty string")
        if not isinstance(service_cls, type) or not issubclass(service_cls, BaseService):
            raise TypeError(f"Service '{name}' must be a subclass of BaseService")
        self._services[name] = service_cls

    def get(self, name: str) -> Type[BaseService]:
        try:
            return self._services[name]
        except KeyError as exc:
            raise KeyError(f"Unknown service '{name}'") from exc

    def create(self, name: str, runtime, **kwargs) -> BaseService:
        try:
            service_cls = self._services[name]
        except KeyError as exc:
            raise KeyError(f"Unknown service '{name}'") from exc
        return service_cls(runtime=runtime, **kwargs)
