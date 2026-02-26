import logging
import threading
import time
from collections.abc import Callable

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from pylsl import StreamInfo, StreamOutlet

from .config import BoardMode, ServerConfig
from .headset import Headset

logger = logging.getLogger(__name__)


class EEGStream:
    """
    Acquires data from an OpenBCI board via BrainFlow and publishes to LSL.

    This is the core primitive — it works independently of the web server.
    Any LSL-compatible tool can consume the stream.
    """

    def __init__(self, config: ServerConfig, headset: Headset | None = None):
        self.config = config
        self.headset = headset
        self._board: BoardShim | None = None
        self._outlet: StreamOutlet | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._error: str | None = None
        self._lock = threading.Lock()

        # Resolved at prepare() time
        self.board_id: int = -1
        self.sampling_rate: int = 0
        self.eeg_channels: list[int] = []
        self.channel_names: list[str] = []
        self.timestamp_channel: int = -1

        # Recording callback — called with each data chunk during recording
        self._recording_callback: Callable[[np.ndarray], None] | None = None

        # Server-side ring buffer for analysis (band power, etc.)
        # Stores full board data rows so any channel can be accessed.
        self._analysis_buffer: np.ndarray | None = None
        self._analysis_write_idx: int = 0
        self._analysis_capacity: int = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def board(self) -> BoardShim | None:
        """Expose board for impedance checker."""
        return self._board

    def prepare(self) -> None:
        """Initialize BrainFlow board and LSL outlet. Does not start streaming."""
        if self.config.board_mode == BoardMode.CYTON:
            if self.headset is None:
                raise ValueError("Headset required for Cyton mode")
            self.board_id = self.headset.board_id
            params = self.headset.get_board_params(self.config.serial_port)
            if not self.config.serial_port:
                raise ValueError(
                    "serial_port is required for Cyton mode. "
                    "On macOS, look for /dev/cu.usbserial-*"
                )
        else:
            self.board_id = BoardIds.SYNTHETIC_BOARD.value
            params = BrainFlowInputParams()

        self._board = BoardShim(self.board_id, params)
        self._board.prepare_session()

        # Resolve board metadata
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)

        # Use headset channel names if available, otherwise default
        if self.headset is not None:
            self.channel_names = self.headset.channel_names[: len(self.eeg_channels)]
        else:
            # Synthetic board — trim to 8 channels
            default_names = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
            self.eeg_channels = self.eeg_channels[:8]
            self.channel_names = default_names[: len(self.eeg_channels)]

        # Create analysis ring buffer (5 seconds of full board data)
        analysis_secs = 5
        self._analysis_capacity = self.sampling_rate * analysis_secs
        num_rows = BoardShim.get_num_rows(self.board_id)
        self._analysis_buffer = np.zeros(
            (num_rows, self._analysis_capacity), dtype=np.float64
        )
        self._analysis_write_idx = 0

        # Create LSL outlet
        self._outlet = self._create_lsl_outlet()

        logger.info(
            "Board prepared: mode=%s, board_id=%d, sr=%d Hz, channels=%s",
            self.config.board_mode.value,
            self.board_id,
            self.sampling_rate,
            self.channel_names,
        )

    def _create_lsl_outlet(self) -> StreamOutlet:
        info = StreamInfo(
            name=self.config.lsl_stream_name,
            type=self.config.lsl_stream_type,
            channel_count=len(self.eeg_channels),
            nominal_srate=float(self.sampling_rate),
            channel_format="float32",
            source_id="brainflow_eeg_stream",
        )

        # Add channel metadata so LSL consumers know which channel is which
        channels_xml = info.desc().append_child("channels")
        for name in self.channel_names:
            ch = channels_xml.append_child("channel")
            ch.append_child_value("label", name)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")

        logger.info("LSL stream created: %s", self.config.lsl_stream_name)
        return StreamOutlet(info, chunk_size=0, max_buffered=360)

    def start(self) -> None:
        """Start acquisition. Calls prepare() if not already done."""
        if self._running:
            return

        if self._board is None:
            self.prepare()

        self._board.start_stream()
        self._running = True
        self._error = None

        self._thread = threading.Thread(
            target=self._acquisition_loop, name="eeg-acquisition", daemon=True
        )
        self._thread.start()
        logger.info("Acquisition started")

    def stop(self) -> None:
        """Stop acquisition and release resources."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._board is not None:
            try:
                self._board.stop_stream()
            except Exception:
                pass
            try:
                self._board.release_session()
            except Exception:
                pass
            self._board = None

        self._outlet = None
        logger.info("Acquisition stopped")

    def pause_stream(self) -> None:
        """Stop the acquisition loop and board stream, but keep the session open."""
        with self._lock:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=2.0)
                self._thread = None
            if self._board is not None:
                self._board.stop_stream()
            logger.info("Stream paused (session still open)")

    def resume_stream(self) -> None:
        """Restart the board stream and acquisition loop."""
        with self._lock:
            if self._board is None:
                raise RuntimeError("No board session to resume")
            self._board.start_stream()
            self._running = True
            self._error = None
            self._thread = threading.Thread(
                target=self._acquisition_loop, name="eeg-acquisition", daemon=True
            )
            self._thread.start()
            logger.info("Stream resumed")

    def start_recording(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register a callback that receives every data chunk during acquisition."""
        self._recording_callback = callback

    def stop_recording(self) -> None:
        """Remove the recording callback."""
        self._recording_callback = None

    def get_recent_data(self, num_samples: int) -> np.ndarray:
        """
        Read the most recent N samples from the server-side analysis buffer.

        Returns a (num_rows, num_samples) array. Does NOT remove data.
        Safe to call from any thread while acquisition is running.
        """
        buf = self._analysis_buffer
        if buf is None:
            return np.zeros((0, 0))

        n = min(num_samples, self._analysis_capacity)
        wi = self._analysis_write_idx

        # Read backward from write index
        if wi >= n:
            return buf[:, wi - n : wi].copy()
        else:
            # Wrap around
            tail = buf[:, self._analysis_capacity - (n - wi) :]
            head = buf[:, :wi]
            return np.hstack([tail, head]).copy()

    def _acquisition_loop(self) -> None:
        """Poll BrainFlow ring buffer and push to LSL + analysis buffer."""
        poll_interval = 0.004  # ~4ms, matches 250 Hz
        while self._running:
            try:
                data = self._board.get_board_data()
                if data.shape[1] > 0:
                    eeg = data[self.eeg_channels, :]  # (n_channels, n_samples)
                    timestamps = data[self.timestamp_channel, :]
                    # LSL push_chunk: list of samples, each sample = list of channel values
                    self._outlet.push_chunk(
                        eeg.T.tolist(), timestamps.tolist()
                    )

                    # Forward to recording callback if active
                    if self._recording_callback is not None:
                        try:
                            self._recording_callback(data)
                        except Exception as e:
                            logger.warning("Recording callback error: %s", e)

                    # Write to analysis ring buffer
                    n = data.shape[1]
                    buf = self._analysis_buffer
                    wi = self._analysis_write_idx
                    cap = self._analysis_capacity

                    if n <= cap - wi:
                        buf[:, wi : wi + n] = data
                        self._analysis_write_idx = wi + n
                    else:
                        # Wrap around
                        first = cap - wi
                        buf[:, wi:] = data[:, :first]
                        rest = n - first
                        buf[:, :rest] = data[:, first:]
                        self._analysis_write_idx = rest
            except Exception as e:
                logger.error("Acquisition error: %s", e)
                self._error = str(e)
                self._running = False
                break

            time.sleep(poll_interval)
