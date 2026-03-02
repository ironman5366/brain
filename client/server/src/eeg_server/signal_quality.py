"""Per-channel EEG signal quality analysis for calibration."""

import numpy as np
from scipy.signal import welch


def analyze_signal_quality(
    data: np.ndarray,
    eeg_channels: list[int],
    channel_names: list[str],
    sampling_rate: int,
    line_freq: float = 60.0,
) -> list[dict]:
    """
    Analyze each channel's signal quality from raw board data.

    Args:
        data: Full board data (num_rows, num_samples) from EEGStream.get_recent_data()
        eeg_channels: Row indices for EEG channels
        channel_names: Human-readable channel names (matching eeg_channels order)
        sampling_rate: Sampling rate in Hz
        line_freq: Power line frequency (60 Hz for US, 50 Hz for EU)

    Returns list of per-channel dicts with metrics and PSD data for visualization.
    """
    results = []
    for i, ch_row in enumerate(eeg_channels):
        ch_data = data[ch_row, :].astype(np.float64)
        result = _analyze_channel(ch_data, channel_names[i], sampling_rate, line_freq)
        results.append(result)
    return results


def _analyze_channel(
    ch_data: np.ndarray,
    name: str,
    sr: int,
    line_freq: float,
) -> dict:
    issues: list[str] = []

    # RMS noise level
    rms_uv = float(np.sqrt(np.mean(ch_data**2)))

    # Welch PSD for spectral analysis
    nperseg = min(len(ch_data), sr * 2)
    freqs, psd = welch(ch_data, fs=sr, nperseg=nperseg)

    # Truncate to 0-60 Hz for visualization payload
    freq_mask = freqs <= 62  # slightly above 60 to include the bin
    vis_freqs = freqs[freq_mask]
    vis_psd = psd[freq_mask]

    # Convert to dB for visualization
    vis_psd_db = 10.0 * np.log10(np.maximum(vis_psd, 1e-12))

    # Line noise: power at line_freq vs broadband (5-45 Hz)
    line_mask = np.abs(freqs - line_freq) < 2.0
    broadband_mask = (freqs >= 5) & (freqs <= 45)
    line_power = float(np.mean(psd[line_mask])) if np.any(line_mask) else 0.0
    broadband_power = float(np.mean(psd[broadband_mask])) if np.any(broadband_mask) else 1e-12
    line_noise_db = float(10 * np.log10(max(line_power, 1e-12) / max(broadband_power, 1e-12)))

    # DC drift / low-frequency noise (< 1 Hz)
    dc_mask = freqs < 1.0
    dc_power = float(np.mean(psd[dc_mask])) if np.any(dc_mask) else 0.0
    dc_drift_uv = float(np.sqrt(dc_power))

    # Alpha detection (8-13 Hz)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    total_mask = (freqs >= 1) & (freqs <= 45)
    alpha_power = float(np.sum(psd[alpha_mask])) if np.any(alpha_mask) else 0.0
    total_power = float(np.sum(psd[total_mask])) if np.any(total_mask) else 1e-12
    alpha_ratio = alpha_power / total_power
    has_alpha = alpha_ratio > 0.15

    # Classify issues
    if rms_uv > 100:
        issues.append("high_noise")
    elif rms_uv < 2:
        issues.append("flat_signal")
    if line_noise_db > 10:
        issues.append("high_line_noise")
    if dc_drift_uv > 50:
        issues.append("dc_drift")

    # Overall rating
    if not issues:
        rating = "good"
    elif len(issues) == 1 and issues[0] == "high_line_noise" and line_noise_db < 20:
        rating = "ok"
    else:
        rating = "bad"

    return {
        "name": name,
        "rms_uv": round(rms_uv, 1),
        "line_noise_db": round(line_noise_db, 1),
        "dc_drift_uv": round(dc_drift_uv, 1),
        "has_alpha": has_alpha,
        "alpha_power_ratio": round(alpha_ratio, 3),
        "rating": rating,
        "issues": issues,
        "psd_frequencies": [round(f, 2) for f in vis_freqs.tolist()],
        "psd_db": [round(v, 2) for v in vis_psd_db.tolist()],
    }
