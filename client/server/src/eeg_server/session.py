"""Recording session management — accumulates EEG data, markers, and responses."""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .board import EEGStream

logger = logging.getLogger(__name__)


@dataclass
class EventMarker:
    """A timestamped event from the frontend."""

    code: str
    timestamp: float  # performance.now() from frontend (ms)
    server_timestamp: float = 0.0  # time.time() on receipt
    block_id: str | None = None
    trial_index: int | None = None
    metadata: dict | None = None


@dataclass
class UserResponse:
    """A user interaction captured during a trial."""

    block_id: str
    trial_index: int
    response_key: str
    reaction_time_ms: float
    timestamp: float  # performance.now()
    correct: bool | None = None


@dataclass
class RecordingSession:
    """
    Manages a single recording session.

    Accumulates EEG data chunks + event markers + user responses.
    On stop(), saves everything to disk.
    """

    session_id: str
    protocol_id: str
    protocol_version: str
    started_at: float

    markers: list[EventMarker] = field(default_factory=list)
    responses: list[UserResponse] = field(default_factory=list)
    _eeg_chunks: list[np.ndarray] = field(default_factory=list)
    _timestamp_chunks: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def create(cls, protocol_id: str, protocol_version: str) -> "RecordingSession":
        session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        return cls(
            session_id=session_id,
            protocol_id=protocol_id,
            protocol_version=protocol_version,
            started_at=time.time(),
        )

    def record_chunk(self, data: np.ndarray, eeg_channels: list[int], timestamp_channel: int) -> None:
        """Called from the EEGStream recording callback with each data chunk."""
        eeg = data[eeg_channels, :]
        timestamps = data[timestamp_channel, :]
        self._eeg_chunks.append(eeg)
        self._timestamp_chunks.append(timestamps)

    def add_markers(self, markers: list[dict]) -> None:
        """Add a batch of markers from the frontend."""
        server_time = time.time()
        for m in markers:
            self.markers.append(
                EventMarker(
                    code=m["code"],
                    timestamp=m["timestamp"],
                    server_timestamp=server_time,
                    block_id=m.get("block_id"),
                    trial_index=m.get("trial_index"),
                    metadata=m.get("metadata"),
                )
            )

    def add_response(self, response: dict) -> None:
        """Add a user response from the frontend."""
        self.responses.append(
            UserResponse(
                block_id=response["block_id"],
                trial_index=response["trial_index"],
                response_key=response["response_key"],
                reaction_time_ms=response["reaction_time_ms"],
                timestamp=response["timestamp"],
                correct=response.get("correct"),
            )
        )

    def save(self, output_dir: Path, channel_names: list[str], sampling_rate: int) -> Path:
        """Save session to disk. Returns the session directory path."""
        duration = time.time() - self.started_at
        session_dir = output_dir / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Concatenate EEG chunks
        if self._eeg_chunks:
            raw_eeg = np.hstack(self._eeg_chunks)
            timestamps = np.concatenate(self._timestamp_chunks)
        else:
            raw_eeg = np.zeros((len(channel_names), 0))
            timestamps = np.array([])

        # Save raw EEG data
        np.savez_compressed(
            session_dir / "eeg_raw.npz",
            eeg=raw_eeg,
            timestamps=timestamps,
            channel_names=channel_names,
            sampling_rate=sampling_rate,
        )

        # Build session metadata
        meta = {
            "session_id": self.session_id,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "started_at": self.started_at,
            "duration_sec": duration,
            "recording": {
                "sampling_rate": sampling_rate,
                "channel_names": channel_names,
                "total_samples": raw_eeg.shape[1],
            },
            "markers": [
                {
                    "code": m.code,
                    "timestamp": m.timestamp,
                    "server_timestamp": m.server_timestamp,
                    "block_id": m.block_id,
                    "trial_index": m.trial_index,
                    "metadata": m.metadata,
                }
                for m in self.markers
            ],
            "responses": [
                {
                    "block_id": r.block_id,
                    "trial_index": r.trial_index,
                    "response_key": r.response_key,
                    "reaction_time_ms": r.reaction_time_ms,
                    "timestamp": r.timestamp,
                    "correct": r.correct,
                }
                for r in self.responses
            ],
            "total_markers": len(self.markers),
            "total_responses": len(self.responses),
        }

        with open(session_dir / "session.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "Session %s saved: %.1fs, %d samples, %d markers",
            self.session_id,
            duration,
            raw_eeg.shape[1],
            len(self.markers),
        )
        return session_dir


class SessionManager:
    """Manages the active recording session and stored sessions."""

    def __init__(self, sessions_dir: Path, eeg_stream: EEGStream):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.eeg_stream = eeg_stream
        self._active: RecordingSession | None = None

    @property
    def active_session(self) -> RecordingSession | None:
        return self._active

    def start_session(self, protocol_id: str, protocol_version: str) -> RecordingSession:
        """Start a new recording session. Stops any stale session first."""
        if self._active is not None:
            logger.warning("Stopping stale session %s", self._active.session_id)
            try:
                self.stop_session()
            except Exception:
                # Force cleanup even if save fails
                self.eeg_stream.stop_recording()
                self._active = None

        session = RecordingSession.create(protocol_id, protocol_version)
        self._active = session

        # Wire up EEG recording callback
        eeg_channels = self.eeg_stream.eeg_channels
        ts_channel = self.eeg_stream.timestamp_channel
        self.eeg_stream.start_recording(
            lambda data: session.record_chunk(data, eeg_channels, ts_channel)
        )

        logger.info("Session started: %s (protocol=%s)", session.session_id, protocol_id)
        return session

    def stop_session(self) -> RecordingSession:
        """Stop the active session, save to disk, and return it."""
        if self._active is None:
            raise RuntimeError("No active session")

        self.eeg_stream.stop_recording()
        session = self._active
        session.save(
            self.sessions_dir,
            self.eeg_stream.channel_names,
            self.eeg_stream.sampling_rate,
        )
        self._active = None
        return session

    def list_sessions(self) -> list[dict]:
        """List all saved sessions with basic metadata."""
        sessions = []
        for session_dir in sorted(self.sessions_dir.iterdir(), reverse=True):
            meta_path = session_dir / "session.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                sessions.append(
                    {
                        "session_id": meta["session_id"],
                        "protocol_id": meta["protocol_id"],
                        "status": "completed",
                        "duration_sec": meta["duration_sec"],
                        "total_markers": meta["total_markers"],
                        "started_at": meta["started_at"],
                        "has_report": (session_dir / "report.md").exists(),
                    }
                )
        return sessions

    def get_session(self, session_id: str) -> dict:
        """Load full session metadata."""
        meta_path = self.sessions_dir / session_id / "session.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        with open(meta_path) as f:
            return json.load(f)

    def get_report(self, session_id: str) -> str | None:
        """Load a session's markdown report, or None if it doesn't exist."""
        report_path = self.sessions_dir / session_id / "report.md"
        if not report_path.exists():
            return None
        return report_path.read_text()

    def save_report(self, session_id: str, content: str) -> None:
        """Write a markdown report for a session."""
        session_dir = self.sessions_dir / session_id
        if not (session_dir / "session.json").exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        (session_dir / "report.md").write_text(content)
