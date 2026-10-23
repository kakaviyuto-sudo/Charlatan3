from __future__ import annotations

import time
from typing import Dict, Optional

from app.lifecycle import LifecycleManager
from config.settings import Settings
from core.event_bus import EventBus
from core.logger import get_logger
from core.state import AppState


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = AppState.from_settings(settings)
        self.logger = get_logger("runtime")
        self.event_bus = EventBus()
        self.lifecycle = LifecycleManager(self.state)

    def register_service(self, name: str, instance: object) -> None:
        if instance is None:
            raise ValueError(f"Cannot register service '{name}' with a None instance")
        self.state.register_service(name, instance)
        self.lifecycle.register(instance)

    def register_services(self, services: Dict[str, object]) -> None:
        for name, instance in services.items():
            self.register_service(name, instance)

    def start(self) -> None:
        self.logger.info("Runtime starting")
        self.lifecycle.start_all()

    def stop(self) -> None:
        self.logger.info("Runtime stopping")
        self.lifecycle.stop_all()
        self.state.shutdown_event.set()

    def run_forever(self, poll_interval: float = 0.1) -> None:
        while not self.state.shutdown_event.is_set():
            self.lifecycle.tick()
            time.sleep(poll_interval)

    def emit(self, event_name: str, payload: object = None) -> None:
        self.event_bus.emit(event_name, payload)
