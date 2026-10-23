from __future__ import annotations

from typing import Dict, Type

from services.base_service import BaseService

class ServiceRegistry:
    def __init__(self):
        self._services: Dict[str, Type[BaseService]] = {}

    def register(self, name: str, service_cls: Type[BaseService]) -> None:
        self._services[name] = service_cls

    def get(self, name: str) -> Type[BaseService]:
        return self._services[name]

    def create(self, name: str, runtime, **kwargs) -> BaseService:
        service_cls = self._services[name]
        return service_cls(runtime=runtime, **kwargs)