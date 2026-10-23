from __future__ import annotations

from typing import Iterable, List

from core.state import AppState
from core.logger import get_logger

class LifecycleManager:
    def __init__(self, state: AppState):
        self.state = state
        self.logger = get_logger("lifecycle")
        self._services: List[object] = []

    def register(self, service: object) -> None:
        if service not in self._services:
            self._services.append(service)

    def register_all(self, services: Iterable[object]) -> None:
        for service in services:
            self.register(service)

    def start_all(self) -> None:
        self.logger.info("Starting all services")
        for service in self._services:
            if hasattr(service, "start"):
                try:
                    service.start()
                    self.state.set_service_status(service.__class__.__name__, "running")
                except Exception as exc:
                    self.logger.exception("Service start failed: %s", service.__class__.__name__)
                    self.state.set_service_status(service.__class__.__name__, "failed")
                    continue

    def stop_all(self) -> None:
        self.logger.info("Stopping all services")
        for service in reversed(self._services):
            if hasattr(service, "stop"):
                try:
                    service.stop()
                    self.state.set_service_status(service.__class__.__name__, "stopped")
                except Exception as exc:
                    self.logger.exception("Service stop failed: %s", service.__class__.__name__)
                    self.state.set_service_status(service.__class__.__name__, "failed")

    def tick(self) -> None:
        for service in self._services:
            if hasattr(service, "tick"):
                try:
                    service.tick()
                except Exception as exc:
                    self.logger.exception("Service tick failed: %s", service.__class__.__name__)