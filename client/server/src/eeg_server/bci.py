"""BCI Controller — manages real-time P300 speller sessions driven by Claude."""

import asyncio
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt

from .board import EEGStream
from .session import SessionManager

logger = logging.getLogger(__name__)

P300_MATRIX = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X",
    "Y", "Z", "1", "2", "3", "4",
    "5", "6", "7", "8", "9", "_",
]

PROTOCOL_ID = "bci-p300-speller"
PROTOCOL_VERSION = "1.0.0"


@dataclass
class BCIController:
    """Server-side state machine for Claude-driven BCI speller sessions."""

    state: str = "idle"  # idle | ready | flashing | proposing
    session_id: str | None = None
    spelled: str = ""

    _event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _flash_done: asyncio.Event = field(default_factory=asyncio.Event)
    _flash_marker_count: int = 0
    _feedback_result: asyncio.Event = field(default_factory=asyncio.Event)
    _feedback_value: bool | None = None

    async def start(self, session_mgr: SessionManager) -> dict:
        """Start a BCI session — begins EEG recording."""
        if self.state != "idle":
            raise RuntimeError(f"Cannot start: state is {self.state}")

        session = session_mgr.start_session(PROTOCOL_ID, PROTOCOL_VERSION)
        self.session_id = session.session_id
        self.state = "ready"
        self.spelled = ""

        await self._push_event("started", {"session_id": session.session_id})
        logger.info("BCI session started: %s", session.session_id)

        return {"session_id": session.session_id, "state": self.state}

    async def flash(self, sequences: int) -> dict:
        """Tell the UI to run flash sequences. Blocks until UI reports done."""
        if self.state not in ("ready", "proposing"):
            raise RuntimeError(f"Cannot flash: state is {self.state}")

        self.state = "flashing"
        self._flash_done.clear()
        self._flash_marker_count = 0

        await self._push_event("flash", {"sequences": sequences})
        logger.info("BCI flash requested: %d sequences", sequences)

        # Block until UI finishes
        await self._flash_done.wait()
        self.state = "ready"

        return {
            "status": "done",
            "sequences": sequences,
            "marker_count": self._flash_marker_count,
        }

    def flash_complete(self, marker_count: int = 0) -> dict:
        """Called by UI when flashing finishes."""
        self._flash_marker_count = marker_count
        self._flash_done.set()
        return {"ok": True}

    async def propose(self, letter: str, message: str = "") -> dict:
        """Propose a letter to the user. Blocks until they accept or reject."""
        if self.state != "ready":
            raise RuntimeError(f"Cannot propose: state is {self.state}")

        self.state = "proposing"
        self._feedback_result.clear()
        self._feedback_value = None

        await self._push_event("propose", {"letter": letter, "message": message})
        logger.info("BCI propose: %s", letter)

        # Block until user responds
        await self._feedback_result.wait()
        accepted = self._feedback_value

        if accepted:
            self.spelled += letter

        self.state = "ready"
        return {"accepted": accepted, "spelled": self.spelled}

    def submit_feedback(self, accepted: bool) -> dict:
        """Called by UI when user accepts or rejects a proposed letter."""
        self._feedback_value = accepted
        self._feedback_result.set()
        return {"ok": True}

    async def send_message(self, text: str) -> dict:
        """Push a text message to display in the UI."""
        await self._push_event("message", {"text": text})
        return {"ok": True}

    async def play_sound(
        self,
        frequency: int = 440,
        duration_ms: int = 200,
        novel: bool = False,
    ) -> dict:
        """Tell the UI to play a sound."""
        await self._push_event(
            "play_sound",
            {"frequency": frequency, "duration_ms": duration_ms, "novel": novel},
        )
        return {"ok": True}

    async def stop(self, session_mgr: SessionManager) -> dict:
        """Stop the BCI session and save data."""
        if self.state == "idle":
            raise RuntimeError("No active BCI session")

        # Unblock any pending waits (flash or propose that lost their UI)
        self._flash_done.set()
        self._feedback_result.set()

        await self._push_event("stopped", {})

        session = session_mgr.stop_session()
        result = {
            "session_id": session.session_id,
            "spelled": self.spelled,
            "total_markers": len(session.markers),
        }

        self.state = "idle"
        self.session_id = None
        logger.info("BCI session stopped: %s", session.session_id)
        return result

    def status(self, session_mgr: SessionManager) -> dict:
        """Current BCI state and session info."""
        active = session_mgr.active_session
        marker_count = len(active.markers) if active else 0
        flash_markers = 0
        if active:
            flash_markers = sum(
                1 for m in active.markers if m.code == "p300_flash"
            )

        return {
            "state": self.state,
            "session_id": self.session_id,
            "spelled": self.spelled,
            "total_markers": marker_count,
            "flash_markers": flash_markers,
        }

    def get_epochs(
        self,
        session_mgr: SessionManager,
        eeg_stream: EEGStream,
        channels: list[str] | None = None,
        window_start_ms: int = 250,
        window_end_ms: int = 500,
        filter_low: float = 0.5,
        filter_high: float = 30.0,
        artifact_threshold_uv: float = 150.0,
    ) -> dict:
        """Extract epochs from live session data and compute row/col P300 scores."""
        active = session_mgr.active_session
        if active is None:
            raise RuntimeError("No active session")

        ch_names = eeg_stream.channel_names
        sr = eeg_stream.sampling_rate
        if channels is None:
            channels = [c for c in ["C3", "C4", "P7", "P8"] if c in ch_names]
        ch_indices = [ch_names.index(c) for c in channels if c in ch_names]

        if not ch_indices:
            raise RuntimeError(f"None of {channels} found in {ch_names}")

        # Get raw EEG from in-memory chunks
        if not active._eeg_chunks:
            raise RuntimeError("No EEG data recorded yet")

        eeg = np.hstack(active._eeg_chunks)  # (n_channels, n_samples)
        timestamps = np.concatenate(active._timestamp_chunks)

        # Bandpass filter
        nyq = sr / 2
        b, a = butter(4, [filter_low / nyq, filter_high / nyq], btype="band")
        eeg_filt = filtfilt(b, a, eeg, axis=1)

        # Extract epochs around p300_flash markers
        pre_samples = int(0.1 * sr)  # 100ms pre-stimulus
        post_samples = int(0.6 * sr)  # 600ms post-stimulus
        epoch_len = pre_samples + post_samples
        t0 = timestamps[0]
        n_samples = eeg_filt.shape[1]

        win_start_samp = pre_samples + int(window_start_ms * sr / 1000)
        win_end_samp = pre_samples + int(window_end_ms * sr / 1000)

        # Group epochs by row/col
        row_epochs: dict[int, list[np.ndarray]] = {i: [] for i in range(6)}
        col_epochs: dict[int, list[np.ndarray]] = {i: [] for i in range(6)}
        n_rejected = 0
        total_flash = 0

        for m in active.markers:
            if m.code != "p300_flash":
                continue
            if m.metadata is None:
                continue

            total_flash += 1
            sample_idx = int((m.server_timestamp - t0) * sr)
            start = sample_idx - pre_samples
            end = sample_idx + post_samples

            if start < 0 or end > n_samples:
                n_rejected += 1
                continue

            epoch = eeg_filt[:, start:end].copy()

            # Baseline correction
            baseline = epoch[:, :pre_samples].mean(axis=1, keepdims=True)
            epoch -= baseline

            # Artifact rejection
            if np.any(np.abs(epoch) > artifact_threshold_uv):
                n_rejected += 1
                continue

            flash_type = m.metadata.get("flash_type")
            flash_index = m.metadata.get("flash_index")
            if flash_type is None or flash_index is None:
                continue

            if flash_type == "row":
                row_epochs[flash_index].append(epoch)
            elif flash_type == "col":
                col_epochs[flash_index].append(epoch)

        # Compute scores: mean amplitude in P300 window at target channels
        def score_epochs(epochs_dict: dict[int, list[np.ndarray]]) -> dict:
            scores = {}
            counts = {}
            channel_detail = {ch: {} for ch in channels}

            for idx in range(6):
                eps = epochs_dict[idx]
                counts[idx] = len(eps)
                if len(eps) == 0:
                    scores[idx] = 0.0
                    for ch in channels:
                        channel_detail[ch][idx] = 0.0
                    continue

                arr = np.array(eps)  # (n_epochs, n_channels, epoch_len)
                mean_epoch = arr.mean(axis=0)

                ch_scores = []
                for ci, ch in zip(ch_indices, channels):
                    val = float(mean_epoch[ci, win_start_samp:win_end_samp].mean())
                    channel_detail[ch][idx] = round(val, 3)
                    ch_scores.append(val)

                scores[idx] = round(float(np.mean(ch_scores)), 3)

            return scores, counts, channel_detail

        row_scores, row_counts, row_ch_detail = score_epochs(row_epochs)
        col_scores, col_counts, col_ch_detail = score_epochs(col_epochs)

        # Predict letter
        best_row = max(range(6), key=lambda i: row_scores[i])
        best_col = max(range(6), key=lambda i: col_scores[i])
        predicted_letter = P300_MATRIX[best_row * 6 + best_col]

        # Confidence: normalized margin between best and second-best
        def confidence(scores: dict[int, float]) -> float:
            vals = sorted(scores.values(), reverse=True)
            if len(vals) < 2 or vals[0] == 0:
                return 0.0
            return round((vals[0] - vals[1]) / abs(vals[0]) if vals[0] != 0 else 0.0, 3)

        row_conf = confidence(row_scores)
        col_conf = confidence(col_scores)

        return {
            "row_scores": [row_scores[i] for i in range(6)],
            "col_scores": [col_scores[i] for i in range(6)],
            "predicted_letter": predicted_letter,
            "confidence": round((row_conf + col_conf) / 2, 3),
            "best_row": best_row,
            "best_col": best_col,
            "n_epochs": {
                **{f"row_{i}": row_counts[i] for i in range(6)},
                **{f"col_{i}": col_counts[i] for i in range(6)},
            },
            "n_rejected": n_rejected,
            "total_flash_markers": total_flash,
            "channel_detail": {
                ch: {
                    "row_scores": [row_ch_detail[ch][i] for i in range(6)],
                    "col_scores": [col_ch_detail[ch][i] for i in range(6)],
                }
                for ch in channels
            },
        }

    def snapshot(
        self,
        session_mgr: SessionManager,
        eeg_stream: EEGStream,
    ) -> dict:
        """Save raw EEG + markers to a temp .npz file for custom analysis."""
        active = session_mgr.active_session
        if active is None:
            raise RuntimeError("No active session")

        if not active._eeg_chunks:
            raise RuntimeError("No EEG data recorded yet")

        eeg = np.hstack(active._eeg_chunks)
        timestamps = np.concatenate(active._timestamp_chunks)

        # Serialize markers to JSON string
        markers_json = json.dumps([
            {
                "code": m.code,
                "timestamp": m.timestamp,
                "server_timestamp": m.server_timestamp,
                "block_id": m.block_id,
                "trial_index": m.trial_index,
                "metadata": m.metadata,
            }
            for m in active.markers
        ])

        path = tempfile.mktemp(
            prefix=f"bci_{active.session_id}_",
            suffix=".npz",
        )
        np.savez(
            path,
            eeg=eeg,
            timestamps=timestamps,
            channel_names=eeg_stream.channel_names,
            sampling_rate=eeg_stream.sampling_rate,
            markers=markers_json,
        )

        duration = time.time() - active.started_at
        logger.info("BCI snapshot saved: %s (%d samples)", path, eeg.shape[1])
        return {
            "path": path,
            "n_samples": eeg.shape[1],
            "n_channels": eeg.shape[0],
            "n_markers": len(active.markers),
            "duration_sec": round(duration, 1),
        }

    async def events(self):
        """Async generator yielding SSE events for the UI."""
        # Send current state on connect
        yield self._format_sse("state", {"state": self.state, "session_id": self.session_id, "spelled": self.spelled})

        while True:
            event = await self._event_queue.get()
            yield event

    async def _push_event(self, event_type: str, data: dict) -> None:
        await self._event_queue.put(self._format_sse(event_type, data))

    @staticmethod
    def _format_sse(event_type: str, data: dict) -> str:
        payload = json.dumps({"type": event_type, **data})
        return f"data: {payload}\n\n"
