"""
Quick check: is the data coming through the server real or synthetic?
Connects to the LSL stream and prints stats on a few seconds of data.

Real EEG: large values (microvolts scale, typically ±50-500 uV),
  varies with touch/movement, 50/60Hz line noise visible.
Synthetic: small values, smooth sinusoids, no response to physical stimuli.

Usage:
    cd server/
    uv run python scripts/check_live_data.py
"""

import logging
import time

import numpy as np
from pylsl import StreamInlet, resolve_byprop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    logger.info("Resolving LSL stream 'BrainFlow_EEG'...")
    streams = resolve_byprop("name", "BrainFlow_EEG", timeout=5.0)
    if not streams:
        logger.error("No LSL stream found! Is the server running?")
        return

    inlet = StreamInlet(streams[0], max_buflen=360)
    logger.info("Connected to LSL stream")

    # Collect 2 seconds of data
    logger.info("Collecting 2 seconds of data...")
    all_samples = []
    start = time.time()
    while time.time() - start < 2.0:
        samples, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=512)
        if len(timestamps) > 0:
            all_samples.extend(samples)

    if not all_samples:
        logger.error("No samples received!")
        return

    data = np.array(all_samples)  # shape: (n_samples, n_channels)
    logger.info("Got %d samples, %d channels", data.shape[0], data.shape[1])
    logger.info("")

    ch_names = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
    for ch in range(min(data.shape[1], 8)):
        col = data[:, ch]
        name = ch_names[ch] if ch < len(ch_names) else f"ch{ch}"
        logger.info(
            "%s: min=%.1f  max=%.1f  mean=%.1f  std=%.1f  range=%.1f",
            name, col.min(), col.max(), col.mean(), col.std(), col.max() - col.min(),
        )

    logger.info("")
    logger.info("If values are small (< 100) and smooth, it's likely synthetic.")
    logger.info("If values are large (1000s+) with high std, it's likely real EEG.")

    inlet.close_stream()


if __name__ == "__main__":
    main()
