from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    type: str = "event"
    payload: dict = field(default_factory=dict)


@dataclass
class UserSpeechDetected(Event):
    text: str = ""


@dataclass
class RecognizedTextReceived(Event):
    text: str = ""


@dataclass
class AIReplyStarted(Event):
    text: str = ""


@dataclass
class AIReplyChunk(Event):
    chunk: str = ""


@dataclass
class AIReplyComplete(Event):
    text: str = ""


@dataclass
class ScreenChanged(Event):
    current_title: str = ""
    summary: str = ""


@dataclass
class YoutubeLiveCaption(Event):
    text: str = ""


@dataclass
class PlaybackStarted(Event):
    text: str = ""


@dataclass
class PlaybackFinished(Event):
    text: str = ""
