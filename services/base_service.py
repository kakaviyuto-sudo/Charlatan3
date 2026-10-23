from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseService(ABC):
    name: str = "base_service"

    def __init__(self, runtime: Any = None, **kwargs: Any):
        self.runtime = runtime
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    def tick(self) -> None:
        """Optional idle loop for services that need polling."""
        pass
