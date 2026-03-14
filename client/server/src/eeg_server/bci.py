"""BCI Controller — manages real-time P300 speller sessions driven by an agent."""

import asyncio
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from .board import EEGStream
from .p300_classifier import (
    P300EEGNet,
    extract_labeled_epochs,
    load_model,
    preprocess_eeg,
    save_model,
    score_epochs,
    train_p300_classifier,
)
from .session import SessionManager
from .timing import estimate_marker_offset_seconds, marker_epoch_seconds

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
    """Server-side state machine for agent-driven BCI speller sessions."""

    state: str = "idle"  # idle | ready | flashing | proposing
    session_id: str | None = None
    spelled: str = ""

    _subscribers: set[asyncio.Queue] = field(default_factory=set)
    _flash_done: asyncio.Event = field(default_factory=asyncio.Event)
    _flash_marker_count: int = 0
    _feedback_result: asyncio.Event = field(default_factory=asyncio.Event)
    _feedback_value: bool | None = None
    _classifier: P300EEGNet | None = None
    _classifier_metrics: dict | None = None

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

        # Block until UI finishes, with timeout
        try:
            await asyncio.wait_for(self._flash_done.wait(), timeout=120)
        except asyncio.TimeoutError:
            self.state = "ready"
            logger.warning("BCI flash timed out after 120s")
            return {
                "status": "timeout",
                "error": "Browser did not respond within 120s. Is the BCI Speller page open?",
                "connected_clients": len(self._subscribers),
            }

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

        # Block until user responds, with timeout
        try:
            await asyncio.wait_for(self._feedback_result.wait(), timeout=300)
        except asyncio.TimeoutError:
            self.state = "ready"
            logger.warning("BCI propose timed out after 300s")
            return {
                "accepted": False,
                "spelled": self.spelled,
                "error": "User did not respond within 300s",
            }

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
            "connected_clients": len(self._subscribers),
            "has_classifier": self._classifier is not None,
        }

    def load_classifier(self, sessions_dir: Path) -> bool:
        """Try to load a previously trained P300 classifier."""
        model_path = sessions_dir / "p300_model.pt"
        if not model_path.exists():
            return False
        try:
            self._classifier, self._classifier_metrics = load_model(model_path)
            logger.info("P300 classifier loaded: %s", self._classifier_metrics)
            return True
        except Exception as e:
            logger.warning("Failed to load P300 classifier: %s", e)
            return False

    def calibrate(
        self,
        session_mgr: SessionManager,
        eeg_stream: EEGStream,
        session_id: str,
    ) -> dict:
        """Train a P300 classifier from a completed copy-spelling session."""
        sessions_dir = session_mgr.sessions_dir
        session_dir = sessions_dir / session_id

        # Load saved session data
        eeg_path = session_dir / "eeg_raw.npz"
        meta_path = session_dir / "session.json"
        if not eeg_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        data = np.load(eeg_path)
        eeg = data["eeg"]
        timestamps = data["timestamps"]
        sr = int(data["sampling_rate"])
        n_channels = eeg.shape[0]

        with open(meta_path) as f:
            meta = json.load(f)

        markers = meta.get("markers", [])
        flash_markers = [m for m in markers if m.get("code") == "p300_flash"]
        labeled = [m for m in flash_markers if m.get("metadata", {}).get("is_target") is not None]

        if len(labeled) < 50:
            raise RuntimeError(
                f"Not enough labeled markers ({len(labeled)}). "
                "Run a copy-spelling protocol first."
            )

        # Preprocess and extract epochs
        eeg_filtered = preprocess_eeg(eeg, sr)
        epochs, labels = extract_labeled_epochs(eeg_filtered, timestamps, markers, sr)

        # Train
        model, metrics = train_p300_classifier(epochs, labels, n_channels=n_channels)

        # Save
        model_path = sessions_dir / "p300_model.pt"
        save_model(model, model_path, metrics)

        # Set on controller
        self._classifier = model
        self._classifier_metrics = metrics

        return {
            "status": "trained",
            "model_path": str(model_path),
            "session_id": session_id,
            **metrics,
        }

    def get_epochs(
        self,
        session_mgr: SessionManager,
        eeg_stream: EEGStream,
        channels: list[str] | None = None,
        window_start_ms: int = 200,
        window_end_ms: int = 600,
        filter_low: float = 0.5,
        filter_high: float = 15.0,
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

        # 60Hz notch filter (narrow Q=30 to remove line noise without affecting signal)
        b_notch, a_notch = iirnotch(60.0, 30.0, sr)
        eeg_filt = filtfilt(b_notch, a_notch, eeg_filt, axis=1)

        # Extract epochs around p300_flash markers
        pre_samples = int(0.2 * sr)  # 200ms pre-stimulus
        post_samples = int(0.8 * sr)  # 800ms post-stimulus
        epoch_len = pre_samples + post_samples
        n_samples = eeg_filt.shape[1]
        offset = estimate_marker_offset_seconds(active.markers)

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
            unix_time = marker_epoch_seconds(m, offset)
            if unix_time is None:
                n_rejected += 1
                continue
            sample_idx = int(np.searchsorted(timestamps, unix_time))
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

        # --- Scoring ---
        row_counts = {i: len(row_epochs[i]) for i in range(6)}
        col_counts = {i: len(col_epochs[i]) for i in range(6)}

        if self._classifier is not None:
            # Classifier-based scoring: sum P(target) per row/col
            scoring_method = "classifier"
            row_scores: dict[int, float] = {}
            col_scores: dict[int, float] = {}

            for idx in range(6):
                if row_epochs[idx]:
                    # Take post-stimulus portion, apply CAR
                    eps = np.array([e[:, pre_samples:] for e in row_epochs[idx]])
                    eps = eps - eps.mean(axis=1, keepdims=True)  # CAR
                    probs = score_epochs(self._classifier, eps)
                    row_scores[idx] = round(float(probs.sum()), 3)
                else:
                    row_scores[idx] = 0.0

                if col_epochs[idx]:
                    eps = np.array([e[:, pre_samples:] for e in col_epochs[idx]])
                    eps = eps - eps.mean(axis=1, keepdims=True)
                    probs = score_epochs(self._classifier, eps)
                    col_scores[idx] = round(float(probs.sum()), 3)
                else:
                    col_scores[idx] = 0.0

            channel_detail = {}  # not meaningful for classifier scoring
        else:
            # Amplitude-based scoring (fallback)
            scoring_method = "amplitude"

            def _score_amplitude(epochs_dict: dict[int, list[np.ndarray]]) -> tuple:
                scores = {}
                ch_detail = {ch: {} for ch in channels}
                for idx in range(6):
                    eps = epochs_dict[idx]
                    if not eps:
                        scores[idx] = 0.0
                        for ch in channels:
                            ch_detail[ch][idx] = 0.0
                        continue
                    arr = np.array(eps)
                    mean_epoch = arr.mean(axis=0)
                    ch_scores = []
                    for ci, ch in zip(ch_indices, channels):
                        val = float(mean_epoch[ci, win_start_samp:win_end_samp].mean())
                        ch_detail[ch][idx] = round(val, 3)
                        ch_scores.append(val)
                    scores[idx] = round(float(np.mean(ch_scores)), 3)
                return scores, ch_detail

            row_scores, row_ch_detail = _score_amplitude(row_epochs)
            col_scores, col_ch_detail = _score_amplitude(col_epochs)
            channel_detail = {
                ch: {
                    "row_scores": [row_ch_detail[ch][i] for i in range(6)],
                    "col_scores": [col_ch_detail[ch][i] for i in range(6)],
                }
                for ch in channels
            }

        # Predict letter
        best_row = max(range(6), key=lambda i: row_scores[i])
        best_col = max(range(6), key=lambda i: col_scores[i])
        predicted_letter = P300_MATRIX[best_row * 6 + best_col]

        # Confidence: normalized margin between best and second-best
        def confidence(scores_dict: dict[int, float]) -> float:
            vals = sorted(scores_dict.values(), reverse=True)
            if len(vals) < 2 or vals[0] == 0:
                return 0.0
            return round((vals[0] - vals[1]) / abs(vals[0]) if vals[0] != 0 else 0.0, 3)

        row_conf = confidence(row_scores)
        col_conf = confidence(col_scores)

        result = {
            "row_scores": [row_scores[i] for i in range(6)],
            "col_scores": [col_scores[i] for i in range(6)],
            "predicted_letter": predicted_letter,
            "confidence": round((row_conf + col_conf) / 2, 3),
            "best_row": best_row,
            "best_col": best_col,
            "scoring_method": scoring_method,
            "n_epochs": {
                **{f"row_{i}": row_counts[i] for i in range(6)},
                **{f"col_{i}": col_counts[i] for i in range(6)},
            },
            "n_rejected": n_rejected,
            "total_flash_markers": total_flash,
        }
        if channel_detail:
            result["channel_detail"] = channel_detail
        return result

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
                "client_time_ms": m.client_time_ms,
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
        """Async generator yielding SSE events for one SSE client."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            # Send current state on connect
            yield self._format_sse("state", {
                "state": self.state,
                "session_id": self.session_id,
                "spelled": self.spelled,
            })

            while True:
                event = await queue.get()
                yield event
        finally:
            self._subscribers.discard(queue)

    async def _push_event(self, event_type: str, data: dict) -> None:
        formatted = self._format_sse(event_type, data)
        for queue in self._subscribers:
            await queue.put(formatted)

    @staticmethod
    def _format_sse(event_type: str, data: dict) -> str:
        payload = json.dumps({"type": event_type, **data})
        return f"data: {payload}\n\n"
