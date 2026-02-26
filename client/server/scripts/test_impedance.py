"""
Test script to reproduce and debug impedance measurement issues.

Usage:
    cd server/
    uv run python scripts/test_impedance.py
"""

import glob
import logging
import time

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_bandpass_api():
    """Figure out the correct perform_bandpass signature."""
    logger.info("=== Testing bandpass filter API ===")

    sampling_rate = 250
    n_samples = 250
    t = np.arange(n_samples) / sampling_rate
    signal = 100.0 * np.sin(2 * np.pi * 31.2 * t) + np.random.randn(n_samples) * 10.0

    # The error log showed: "Start Freq:31.2 , Stop Freq:4"
    # So BrainFlow wants (start_freq=low_cutoff, stop_freq=high_cutoff), NOT (center, bandwidth)
    # To bandpass around 31.2 Hz with 4 Hz bandwidth: low=29.2, high=33.2

    tests = [
        ("low=29.2, high=33.2, order=2", 29.2, 33.2, 2),
        ("low=27, high=35, order=2", 27.0, 35.0, 2),
        ("low=25, high=37, order=3", 25.0, 37.0, 3),
    ]

    for desc, low, high, order in tests:
        s = signal.copy().astype(np.float64)
        try:
            DataFilter.perform_bandpass(
                s, sampling_rate, low, high, order,
                FilterTypes.BUTTERWORTH.value, 0.0,
            )
            logger.info("  OK: %s -> RMS=%.2f", desc, np.sqrt(np.mean(s**2)))
        except Exception as e:
            logger.info("  FAIL: %s -> %s", desc, e)


def test_with_cyton():
    """Test impedance measurement against the live Cyton board."""
    logger.info("=== Testing with live Cyton board ===")

    params = BrainFlowInputParams()
    ports = glob.glob("/dev/cu.usbserial-*")
    if not ports:
        logger.error("No serial port found")
        return
    params.serial_port = ports[0]
    logger.info("Using serial port: %s", ports[0])

    board_id = BoardIds.CYTON_BOARD.value
    board = BoardShim(board_id, params)
    board.prepare_session()

    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    logger.info("Board ready: sr=%d, eeg_channels=%s", sampling_rate, eeg_channels)

    board.start_stream()
    time.sleep(1.0)

    # Test just channel 0 (Fp1)
    ch_idx = 0
    ch_num = ch_idx + 1
    eeg_row = eeg_channels[ch_idx]

    logger.info("--- Measuring channel %d (Fp1) ---", ch_idx)

    # Enable lead-off on P input
    logger.info("Enabling lead-off: z%d10Z", ch_num)
    board.config_board(f"z{ch_num}10Z")

    time.sleep(0.5)  # settle

    # Flush, then collect
    board.get_board_data()
    time.sleep(1.0)
    data = board.get_board_data()

    # Disable lead-off
    board.config_board(f"z{ch_num}00Z")

    logger.info("Data shape: %s", data.shape)
    channel_data = data[eeg_row, :].astype(np.float64)
    logger.info(
        "Channel data: %d samples, min=%.2f, max=%.2f, mean=%.2f, std=%.2f",
        len(channel_data), channel_data.min(), channel_data.max(),
        channel_data.mean(), np.std(channel_data),
    )

    # Bandpass around 31.2 Hz: use correct API (low_cutoff, high_cutoff)
    low_cutoff = 29.0
    high_cutoff = 33.0
    order = 2

    filtered = channel_data.copy()
    try:
        DataFilter.perform_bandpass(
            filtered, sampling_rate, low_cutoff, high_cutoff, order,
            FilterTypes.BUTTERWORTH.value, 0.0,
        )
        v_rms_uv = np.sqrt(np.mean(filtered**2))
        logger.info("Filtered RMS: %.2f uV", v_rms_uv)

        # Compute impedance
        v_rms_v = v_rms_uv * 1e-6
        lead_off_current_a = 6e-9
        series_resistance = 2200.0
        impedance = (v_rms_v * np.sqrt(2)) / lead_off_current_a - series_resistance
        impedance = max(0.0, impedance)
        logger.info("Impedance: %.1f Ohm (%.1f kOhm)", impedance, impedance / 1000)
    except Exception as e:
        logger.error("Filter still failing: %s", e)

    board.stop_stream()
    board.release_session()
    logger.info("Done.")


if __name__ == "__main__":
    test_bandpass_api()
    print()
    test_with_cyton()
