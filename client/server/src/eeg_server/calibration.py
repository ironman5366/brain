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
    """

    _event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _last_impedance: dict | None = None
    _last_signal: list[dict] | None = None
    _messages: list[str] = field(default_factory=list)

    async def events(self):
        """Async generator yielding SSE events for the calibration UI."""
        # Send current state on connect
        yield self._format_sse("state", self._get_state())

        while True:
            event = await self._event_queue.get()
            yield event

    async def send_message(self, text: str) -> dict:
        """Push a message from Claude to display in the UI."""
        self._messages.append(text)
        if len(self._messages) > MAX_MESSAGES:
            self._messages = self._messages[-MAX_MESSAGES:]
        await self._push_event("message", {"text": text})
        logger.info("Calibration message: %s", text[:80])
        return {"ok": True}

    async def push_impedance_update(self, results: dict) -> None:
        """Cache impedance results and push to SSE stream."""
        self._last_impedance = results
        await self._push_event("impedance", results)

    async def push_signal_update(self, results: list[dict]) -> None:
        """Cache signal quality results and push to SSE stream."""
        self._last_signal = results
        await self._push_event("signal_quality", {"channels": results})

    def _get_state(self) -> dict:
        """Current calibration state for initial SSE connection."""
        return {
            "impedance": self._last_impedance,
            "signal_quality": self._last_signal,
            "messages": self._messages[-20:],
        }

    async def _push_event(self, event_type: str, data: dict | list) -> None:
        await self._event_queue.put(
            self._format_sse(event_type, data if isinstance(data, dict) else {"data": data})
        )

    @staticmethod
    def _format_sse(event_type: str, data: dict) -> str:
        payload = json.dumps({"type": event_type, **data})
        return f"data: {payload}\n\n"
