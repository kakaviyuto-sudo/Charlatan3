from __future__ import annotations

from abc import ABC, abstractmethod

class BaseService(ABC):
    name: str = "base_service"

    def __init__(self, runtime=None):
        self.runtime = runtime

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    def tick(self) -> None:
        """Optional idle loop for services that need polling."""
        pass