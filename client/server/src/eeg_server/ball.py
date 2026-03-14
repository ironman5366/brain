"""Server-side controller for the agent-driven ball paradigm."""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field

from .board import EEGStream
from .control import ControlSignalComputer
from .session import SessionManager

PROTOCOL_ID = "ball-control"
PROTOCOL_VERSION = "1.0.0"
DEFAULT_WINDOW_SEC = 1.0
DEFAULT_TICK_HZ = 20.0
DEFAULT_MAX_SPEED = 0.55  # normalized units / second
DEFAULT_SMOOTHING = 0.35
DEFAULT_MARGIN = 0.04
TRAIL_LENGTH = 40


@dataclass
class BallController:
    """Authoritative server-side state for the ball control paradigm."""

    state: str = "idle"  # idle | running
    session_id: str | None = None
    message: str | None = None
    started_at: float | None = None
    window_sec: float = DEFAULT_WINDOW_SEC
    tick_hz: float = DEFAULT_TICK_HZ
    max_speed: float = DEFAULT_MAX_SPEED
    smoothing: float = DEFAULT_SMOOTHING
    margin: float = DEFAULT_MARGIN

    _x: float = 0.5
    _y: float = 0.5
    _vx: float = 0.0
    _vy: float = 0.0
    _last_tick_monotonic: float | None = None
    _latest_control: dict | None = None
    _target: dict | None = None  # {"x": float, "y": float} or None
    _trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    _subscribers: set[asyncio.Queue] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._trail.append((self._x, self._y))

    async def start(self, session_mgr: SessionManager, control: ControlSignalComputer) -> dict:
        """Start a ball-control run and reset all state."""
        if self.state != "idle":
            raise RuntimeError(f"Cannot start: state is {self.state}")

        session = session_mgr.start_session(PROTOCOL_ID, PROTOCOL_VERSION)
        self.session_id = session.session_id
        self.state = "running"
        self.started_at = time.time()
        self.message = None
        self._reset_state(control)

        payload = self.status()
        await self._broadcast("started", payload)
        return payload

    async def stop(self, session_mgr: SessionManager) -> dict:
        """Stop the active ball-control run and save EEG to disk."""
        if self.state == "idle":
            raise RuntimeError("No active ball run")

        session = session_mgr.stop_session()
        duration_sec = max(0.0, time.time() - session.started_at)
        result = {
            "session_id": session.session_id,
            "duration_sec": round(duration_sec, 1),
            "state": "idle",
        }

        await self._broadcast("stopped", result)

        self.state = "idle"
        self.session_id = None
        self.message = None
        self.started_at = None
        self._clear_ball_state()

        return result

    async def reset(self, control: ControlSignalComputer) -> dict:
        """Recenter the ball and reset adaptive control normalization."""
        self._reset_state(control)
        payload = self.status()
        await self._broadcast("reset", payload)
        return payload

    async def set_target(self, x: float | None, y: float | None) -> dict:
        """Set or clear the target position for the ball UI."""
        if x is None or y is None:
            self._target = None
        else:
            self._target = {"x": round(x, 4), "y": round(y, 4)}
        await self._broadcast("target", {"target": self._target})
        return {"ok": True, "target": self._target}

    async def send_message(self, text: str) -> dict:
        """Push an instruction or observation to the ball UI."""
        self.message = text
        payload = {"text": text, "session_id": self.session_id}
        await self._broadcast("message", payload)
        return {"ok": True}

    def status(self) -> dict:
        """Current ball state, control snapshot, and UI message."""
        return {
            "state": self.state,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "message": self.message,
            "ball": {
                "x": round(self._x, 4),
                "y": round(self._y, 4),
                "vx": round(self._vx, 4),
                "vy": round(self._vy, 4),
                "trail": [
                    {"x": round(x, 4), "y": round(y, 4)}
                    for x, y in self._trail
                ],
            },
            "target": self._target,
            "control": self._latest_control,
            "tick_hz": self.tick_hz,
            "window_sec": self.window_sec,
            "connected_clients": len(self._subscribers),
        }

    async def tick(self, eeg_stream: EEGStream, control: ControlSignalComputer) -> None:
        """Advance the ball using the latest EEG-derived control signal."""
        if self.state != "running" or not eeg_stream.is_running:
            return

        loop = asyncio.get_event_loop()
        num_samples = max(int(eeg_stream.sampling_rate * self.window_sec), 8)
        control_result = await loop.run_in_executor(
            None,
            lambda: control.update(
                eeg_stream.get_recent_data(num_samples),
                eeg_stream.eeg_channels,
                eeg_stream.channel_names,
                eeg_stream.sampling_rate,
            ),
        )

        now = time.monotonic()
        if self._last_tick_monotonic is None:
            self._last_tick_monotonic = now

        dt = min(max(now - self._last_tick_monotonic, 0.0), 0.25)
        self._last_tick_monotonic = now

        self._latest_control = control_result

        if control_result.get("calibrated"):
            target_vx = float(control_result["asymmetry"]) * self.max_speed
            target_vy = -((float(control_result["concentration"]) - 0.5) * 2.0) * self.max_speed

            self._vx += self.smoothing * (target_vx - self._vx)
            self._vy += self.smoothing * (target_vy - self._vy)
        else:
            self._vx *= 0.8
            self._vy *= 0.8

        self._x += self._vx * dt
        self._y += self._vy * dt
        self._apply_bounds()
        self._trail.append((self._x, self._y))

        await self._broadcast("telemetry", self.status())

    async def events(self):
        """Async generator yielding SSE events for one SSE client."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            yield self._format_sse("state", self.status())
            while True:
                event = await queue.get()
                yield event
        finally:
            self._subscribers.discard(queue)

    def _reset_state(self, control: ControlSignalComputer) -> None:
        control.reset()
        self._clear_ball_state()
        self._last_tick_monotonic = time.monotonic()

    def _clear_ball_state(self) -> None:
        self._x = 0.5
        self._y = 0.5
        self._vx = 0.0
        self._vy = 0.0
        self._latest_control = None
        self._target = None
        self._last_tick_monotonic = None
        self._trail.clear()
        self._trail.append((self._x, self._y))

    def _apply_bounds(self) -> None:
        def clamp(value: float) -> float:
            if value < self.margin:
                return self.margin + (value - self.margin) * 0.3
            if value > 1.0 - self.margin:
                return (1.0 - self.margin) + (value - (1.0 - self.margin)) * 0.3
            return value

        self._x = clamp(self._x)
        self._y = clamp(self._y)

    async def _broadcast(self, event_type: str, data: dict) -> None:
        formatted = self._format_sse(event_type, data)
        for queue in self._subscribers:
            await queue.put(formatted)

    @staticmethod
    def _format_sse(event_type: str, data: dict) -> str:
        payload = json.dumps({"type": event_type, **data})
        return f"data: {payload}\n\n"
