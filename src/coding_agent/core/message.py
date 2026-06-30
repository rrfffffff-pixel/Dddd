"""Agent-to-agent messaging."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    FEEDBACK = "feedback"
    STATUS = "status"
    ERROR = "error"


@dataclass
class Message:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    sender: str = ""
    receiver: str = ""
    type: MessageType = MessageType.TASK
    content: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class MessageBus:
    """In-process message bus for agent communication."""

    def __init__(self) -> None:
        self._queues: dict[str, list[Message]] = {}
        self._subscribers: dict[str, list] = {}

    def publish(self, message: Message) -> None:
        queue = self._queues.setdefault(message.receiver, [])
        queue.append(message)
        for callback in self._subscribers.get(message.receiver, []):
            callback(message)

    def subscribe(self, agent_id: str, callback) -> None:
        self._subscribers.setdefault(agent_id, []).append(callback)

    def consume(self, agent_id: str) -> Message | None:
        queue = self._queues.get(agent_id, [])
        return queue.pop(0) if queue else None

    def peek(self, agent_id: str) -> list[Message]:
        return list(self._queues.get(agent_id, []))

    def clear(self, agent_id: str) -> None:
        self._queues.pop(agent_id, None)
