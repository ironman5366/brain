"""MCP server for Claude to control the BCI speller.

Thin wrapper around the FastAPI BCI endpoints. Each tool makes an HTTP call
to the running EEG server and returns the JSON response.

Usage:
    uv run python -m eeg_server.mcp_server          # stdio transport (default)
"""

import json
import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="brain-bci")

BASE_URL = os.environ.get("EEG_SERVER_URL", "http://localhost:8765")


def _sync_post(path: str, body: dict | None = None) -> str:
    """Synchronous POST to the EEG server. Used for non-blocking endpoints."""
    with httpx.Client(base_url=BASE_URL, timeout=10) as c:
        r = c.post(path, json=body or {})
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)


def _sync_get(path: str, params: dict | None = None) -> str:
    """Synchronous GET to the EEG server."""
    with httpx.Client(base_url=BASE_URL, timeout=10) as c:
        r = c.get(path, params=params or {})
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)


def _sync_post_blocking(path: str, body: dict | None = None, timeout: float = 300) -> str:
    """Synchronous POST that may block for a long time (flash, propose)."""
    with httpx.Client(base_url=BASE_URL, timeout=timeout) as c:
        r = c.post(path, json=body or {})
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)


@mcp.tool()
def bci_start() -> str:
    """Start a BCI speller session.

    Begins EEG recording and shows the P300 matrix in the user's browser.
    The user should have the BCI Speller page open and their headset on.
    """
    return _sync_post("/api/bci/start")


@mcp.tool()
def bci_flash(sequences: int = 5) -> str:
    """Run P300 flash sequences on the 6x6 letter matrix.

    All 6 rows and 6 columns flash in random order, repeated for the given
    number of sequences. Each sequence = 12 flashes (6 rows + 6 cols).
    The user should be attending to their chosen letter during flashing.

    Blocks until the browser finishes all flash sequences.

    Args:
        sequences: Number of flash sequences to run (default 5). More sequences
            = more data = better accuracy, but takes longer.
    """
    return _sync_post_blocking("/api/bci/flash", {"sequences": sequences})


@mcp.tool()
def bci_epochs(
    channels: str = "C3,C4,P7,P8",
    window_start_ms: int = 250,
    window_end_ms: int = 500,
    filter_low: float = 0.5,
    filter_high: float = 30.0,
    artifact_threshold_uv: float = 150.0,
) -> str:
    """Analyze P300 epochs and get row/column scores for letter prediction.

    Extracts epochs around each flash marker, bandpass filters, baseline
    corrects, rejects artifacts, then scores each row and column by mean
    amplitude in the P300 window. The row and column with the highest scores
    intersect at the predicted letter.

    Returns: row_scores (6 floats), col_scores (6 floats), predicted_letter,
    confidence, per-channel detail, epoch counts, rejection count.

    Args:
        channels: Comma-separated channel names to use for scoring.
        window_start_ms: Start of P300 window in ms post-stimulus.
        window_end_ms: End of P300 window in ms post-stimulus.
        filter_low: Bandpass filter low cutoff in Hz.
        filter_high: Bandpass filter high cutoff in Hz.
        artifact_threshold_uv: Reject epochs exceeding this amplitude (uV).
    """
    return _sync_get(
        "/api/bci/epochs",
        {
            "channels": channels,
            "window_start_ms": window_start_ms,
            "window_end_ms": window_end_ms,
            "filter_low": filter_low,
            "filter_high": filter_high,
            "artifact_threshold_uv": artifact_threshold_uv,
        },
    )


@mcp.tool()
def bci_snapshot() -> str:
    """Save raw EEG data + markers from the current session to a .npz file.

    Returns the file path. Load with numpy for fully custom analysis:

        data = np.load(path, allow_pickle=True)
        eeg = data['eeg']              # (n_channels, n_samples)
        timestamps = data['timestamps'] # (n_samples,)
        ch_names = list(data['channel_names'])
        sr = int(data['sampling_rate'])
        markers = json.loads(str(data['markers']))

    Each marker has: code, timestamp, server_timestamp, metadata (with
    flash_type, flash_index, sequence_num for p300_flash markers).
    """
    return _sync_post("/api/bci/snapshot")


@mcp.tool()
def bci_propose(letter: str, message: str = "") -> str:
    """Propose a letter to the user.

    Shows the letter prominently in the UI with an optional message.
    Blocks until the user clicks Accept or Reject.

    Returns: {accepted: bool, spelled: str} where spelled is the
    accumulated string of all accepted letters so far.

    Args:
        letter: The letter to propose (single character from the P300 matrix).
        message: Optional message to display alongside the proposal
            (e.g., "I'm fairly confident this is B based on strong P8 response").
    """
    return _sync_post_blocking("/api/bci/propose", {"letter": letter, "message": message})


@mcp.tool()
def bci_message(text: str) -> str:
    """Show a message to the user in the BCI interface.

    Use this to communicate status, instructions, or analysis findings.

    Args:
        text: The message text to display.
    """
    return _sync_post("/api/bci/message", {"text": text})


@mcp.tool()
def bci_play_sound(frequency: int = 440, duration_ms: int = 200, novel: bool = False) -> str:
    """Play a sound in the user's browser.

    Args:
        frequency: Tone frequency in Hz (ignored if novel=True).
        duration_ms: Duration in milliseconds.
        novel: If True, plays a random novel synthesized sound instead of a pure tone.
    """
    return _sync_post(
        "/api/bci/play-sound",
        {"frequency": frequency, "duration_ms": duration_ms, "novel": novel},
    )


@mcp.tool()
def bci_status() -> str:
    """Get current BCI session status.

    Returns: state (idle/ready/flashing/proposing), session_id,
    spelled text so far, marker counts.
    """
    return _sync_get("/api/bci/status")


@mcp.tool()
def bci_stop() -> str:
    """Stop the BCI session and save all data to disk.

    Returns: session_id, spelled text, total marker count.
    """
    return _sync_post("/api/bci/stop")


# --- Calibration Tools ---


@mcp.tool()
def calibration_check_impedance() -> str:
    """Check electrode impedance on all 8 channels.

    Measures skin-electrode contact quality using lead-off detection.
    Takes ~15 seconds. The EEG stream pauses briefly during measurement.

    Returns per-channel impedance with wire colors for physical identification:
    - Each channel: name, wire_color, pin, impedance_kohms, rating
    - Ratings: "good" (<50 kOhm), "ok" (<200 kOhm), "bad" (>=200 kOhm)
    - all_good: true if every channel is "good"

    Wire colors (Ultracortex Mark IV):
    Fp1=grey, Fp2=purple, C3=blue, C4=green,
    P7=yellow, P8=orange, O1=red, O2=brown
    Ear clips (SRB2 + BIAS) = black
    """
    return _sync_post_blocking("/api/calibration/check-impedance", timeout=30)


@mcp.tool()
def calibration_check_signal(duration_sec: float = 3.0) -> str:
    """Analyze live EEG signal quality on all channels.

    Reads recent EEG data from the ring buffer and computes per-channel metrics.
    Does NOT pause the stream. Fast (~instant).

    Per-channel metrics:
    - rms_uv: RMS amplitude in microvolts (good: 10-50, flat: <2, noisy: >100)
    - line_noise_db: 60 Hz power relative to broadband (good: <10 dB)
    - dc_drift_uv: Low-frequency drift (good: <50 uV)
    - has_alpha: Whether alpha rhythm (8-13 Hz) is detectable
    - rating: "good", "ok", or "bad"
    - issues: List of specific problems (high_noise, flat_signal, high_line_noise, dc_drift)
    - wire_color: Physical wire color for this electrode
    - psd_frequencies, psd_db: Per-channel power spectrum (0-60 Hz) for visualization

    Wire colors: Fp1=grey, Fp2=purple, C3=blue, C4=green,
    P7=yellow, P8=orange, O1=red, O2=brown

    Args:
        duration_sec: Seconds of recent EEG data to analyze (default 3.0).
    """
    return _sync_get("/api/calibration/check-signal", {"duration_sec": duration_sec})


@mcp.tool()
def calibration_message(text: str) -> str:
    """Show a message to the user in the calibration UI.

    Use this to give specific instructions about adjusting electrodes.
    Always reference the wire color and electrode name together,
    e.g. "Push down on the GREY wire (Fp1) and wiggle it slightly."

    Args:
        text: The instruction text to display prominently in the UI.
    """
    return _sync_post("/api/calibration/message", {"text": text})


@mcp.tool()
def calibration_status() -> str:
    """Get a summary of the current calibration state.

    Returns the most recent impedance results, signal quality results,
    and the last few messages sent to the user.
    """
    return _sync_get("/api/calibration/status")


if __name__ == "__main__":
    mcp.run(transport="stdio")
