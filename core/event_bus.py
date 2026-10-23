from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, List

class EventBus:
    def __init__(self):
        self._listeners: DefaultDict[str, List[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_name: str, callback: Callable[[Any], None]) -> None:
        self._listeners[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable[[Any], None]) -> None:
        if event_name in self._listeners:
            self._listeners[event_name] = [
                fn for fn in self._listeners[event_name] if fn is not callback
            ]

    def emit(self, event_name: str, payload: Any = None) -> None:
        for callback in list(self._listeners.get(event_name, [])):
            callback(payload)