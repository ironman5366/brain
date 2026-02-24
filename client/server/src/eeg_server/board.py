import logging
import threading
import time

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

    def _acquisition_loop(self) -> None:
        """Poll BrainFlow ring buffer and push to LSL."""
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
            except Exception as e:
                logger.error("Acquisition error: %s", e)
                self._error = str(e)
                self._running = False
                break

            time.sleep(poll_interval)
