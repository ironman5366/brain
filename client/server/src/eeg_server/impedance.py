"""Electrode impedance measurement via ADS1299 lead-off detection."""

import logging
import random
import time
from collections.abc import Callable

import numpy as np
from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, FilterTypes

from .headset import Headset

logger = logging.getLogger(__name__)

SETTLE_TIME_SEC = 0.5
MEASURE_TIME_SEC = 1.0


class ImpedanceChecker:
    """Measures electrode impedance using lead-off detection on a real board."""

    def __init__(
        self,
        board: BoardShim,
        board_id: int,
        eeg_channels: list[int],
        headset: Headset,
    ):
        self.board = board
        self.board_id = board_id
        self.eeg_channels = eeg_channels
        self.headset = headset
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)

    def measure_all(
        self, callback: Callable[[int, float], None] | None = None
    ) -> dict[int, float]:
        """
        Measure impedance on all channels sequentially.

        Returns {channel_index: impedance_ohms}.
        callback(channel_index, impedance_ohms) is called after each channel.
        """
        results: dict[int, float] = {}
        for ch_idx in range(len(self.eeg_channels)):
            z = self._measure_channel(ch_idx)
            results[ch_idx] = z
            logger.info(
                "Channel %d impedance: %.1f kOhm", ch_idx, z / 1000
            )
            if callback:
                callback(ch_idx, z)

        # Reset all channels to default settings (gain=24x, etc.)
        # The z commands can leave ADS1299 registers in a bad state.
        self.board.config_board("d")
        time.sleep(0.1)

        return results

    def _measure_channel(self, ch_idx: int) -> float:
        ch_num = ch_idx + 1  # Cyton SDK uses 1-indexed channels

        # Enable lead-off on P input
        self.board.config_board(f"z{ch_num}10Z")

        # Wait for settling
        time.sleep(SETTLE_TIME_SEC)

        # Flush stale data, then collect fresh samples
        self.board.get_board_data()
        time.sleep(MEASURE_TIME_SEC)
        data = self.board.get_board_data()

        # Disable lead-off
        self.board.config_board(f"z{ch_num}00Z")

        if data.shape[1] == 0:
            logger.warning("No data collected for channel %d", ch_idx)
            return float("inf")

        # Extract channel data
        eeg_row = self.eeg_channels[ch_idx]
        channel_data = data[eeg_row, :].astype(np.float64)

        # Bandpass filter around the lead-off frequency
        # BrainFlow perform_bandpass takes (low_cutoff, high_cutoff), NOT (center, bandwidth)
        freq = self.headset.lead_off_freq_hz
        half_bw = 2.0  # ±2 Hz around lead-off frequency
        DataFilter.perform_bandpass(
            channel_data,
            self.sampling_rate,
            freq - half_bw,  # low cutoff
            freq + half_bw,  # high cutoff
            2,               # filter order
            FilterTypes.BUTTERWORTH.value,
            0.0,             # ripple (unused for Butterworth)
        )

        # Compute RMS voltage (BrainFlow returns microvolts for Cyton)
        v_rms_uv = np.sqrt(np.mean(channel_data**2))
        v_rms_v = v_rms_uv * 1e-6

        # Z = V_peak / I_peak = V_rms * sqrt(2) / I_peak
        # I_peak = lead_off_current_a (6 nA for ADS1299)
        current_a = self.headset.lead_off_current_a
        impedance = (v_rms_v * np.sqrt(2)) / current_a

        # Subtract series resistance on the board
        impedance -= self.headset.series_resistance_ohm

        return max(0.0, float(impedance))


class SyntheticImpedanceChecker:
    """Returns simulated impedance values for development without hardware."""

    def __init__(self, num_channels: int):
        self.num_channels = num_channels

    def measure_all(
        self, callback: Callable[[int, float], None] | None = None
    ) -> dict[int, float]:
        results: dict[int, float] = {}
        for ch_idx in range(self.num_channels):
            z = self._measure_channel(ch_idx)
            results[ch_idx] = z
            if callback:
                callback(ch_idx, z)
        return results

    def _measure_channel(self, ch_idx: int) -> float:
        # Simulate realistic range: mix of good, ok, and bad contacts
        time.sleep(0.3)
        return random.uniform(5_000, 300_000)
