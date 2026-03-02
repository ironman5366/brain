"""Calibration controller — manages Claude-driven headset calibration sessions."""

import asyncio
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_MESSAGES = 50


@dataclass
class CalibrationController:
    """
    Server-side controller for Claude-driven EEG calibration.

    Claude calls non-blocking MCP tools. This controller:
    - Caches the latest impedance and signal quality results
    - Pushes events to the calibration UI via SSE
    - Maintains a message history of Claude's instructions

    Uses a subscriber set so multiple SSE clients each get every event.
    """

    _subscribers: set[asyncio.Queue] = field(default_factory=set)
    _last_impedance: dict | None = None
    _last_signal: list[dict] | None = None
    _messages: list[str] = field(default_factory=list)

    async def events(self):
        """Async generator yielding SSE events for one SSE client."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            # Send current state on connect
            yield self._format_sse("state", self._get_state())

            while True:
                event = await queue.get()
                yield event
        finally:
            self._subscribers.discard(queue)

    async def send_message(self, text: str) -> dict:
        """Push a message from Claude to display in the UI."""
        self._messages.append(text)
        if len(self._messages) > MAX_MESSAGES:
            self._messages = self._messages[-MAX_MESSAGES:]
        await self._broadcast("message", {"text": text})
        logger.info("Calibration message: %s", text[:80])
        return {"ok": True}

    async def push_impedance_update(self, results: dict) -> None:
        """Cache impedance results and push to all SSE clients."""
        self._last_impedance = results
        await self._broadcast("impedance", results)

    async def push_signal_update(self, results: list[dict]) -> None:
        """Cache signal quality results and push to all SSE clients."""
        self._last_signal = results
        await self._broadcast("signal_quality", {"channels": results})

    def _get_state(self) -> dict:
        """Current calibration state for initial SSE connection."""
        return {
            "impedance": self._last_impedance,
            "signal_quality": self._last_signal,
            "messages": self._messages[-20:],
        }

    async def _broadcast(self, event_type: str, data: dict | list) -> None:
        """Push an SSE event to all connected subscribers."""
        formatted = self._format_sse(
            event_type, data if isinstance(data, dict) else {"data": data}
        )
        for queue in self._subscribers:
            await queue.put(formatted)

    @staticmethod
    def _format_sse(event_type: str, data: dict) -> str:
        payload = json.dumps({"type": event_type, **data})
        return f"data: {payload}\n\n"
