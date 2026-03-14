"""Real-time control signal computation for brain-controlled ball movement.

Computes two control signals from EEG:
- Alpha asymmetry (left/right): covert attention lateralization from P7/O1 vs P8/O2
- Concentration (up/down): beta/alpha power ratio from C3/C4/P7/P8

Uses adaptive normalization (running EMA of mean + std) so no explicit calibration needed.
"""

import threading

import numpy as np
from scipy.signal import welch

# Channels used for each control signal
ASYMMETRY_LEFT = ("P7", "O1")
ASYMMETRY_RIGHT = ("P8", "O2")
CONCENTRATION_CHANNELS = ("C3", "C4", "P7", "P8")

ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
LINE_NOISE_BAND_HZ = 60.0
MIN_RMS_UV = 2.0
MAX_RMS_UV = 500.0
MAX_LINE_NOISE_DB = 40.0
MAX_DC_DRIFT_UV = 200.0

class ControlSignalComputer:
    """Stateful controller that maintains adaptive normalization."""

    def __init__(self, smooth_alpha: float = 0.3, norm_alpha: float = 0.05):
        self._smooth_alpha = smooth_alpha
        self._norm_alpha = norm_alpha
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Reset adaptive normalization and smoothed outputs."""
        with self._lock:
            self._update_count = 0

            # Smoothed output values
            self._smooth_asym = 0.0
            self._smooth_conc = 0.5

            # Running stats for adaptive normalization
            self._asym_mean = 0.0
            self._asym_var = 0.01
            self._conc_mean = 1.0
            self._conc_var = 0.1

            # Last good values for artifact hold
            self._last_asym = 0.0
            self._last_conc = 0.5

    def update(
        self,
        data: np.ndarray,
        eeg_channels: list[int],
        channel_names: list[str],
        sampling_rate: int,
    ) -> dict:
        with self._lock:
            return self._update_locked(
                data,
                eeg_channels,
                channel_names,
                sampling_rate,
            )

    def _update_locked(
        self,
        data: np.ndarray,
        eeg_channels: list[int],
        channel_names: list[str],
        sampling_rate: int,
    ) -> dict:
        channel_powers = _compute_channel_powers(
            data, eeg_channels, channel_names, sampling_rate
        )

        if not channel_powers:
            return self._hold_result(channel_powers)

        # Compute raw control signals
        raw_asym = _compute_asymmetry(channel_powers)
        raw_conc = _compute_concentration(channel_powers)

        if raw_asym is None or raw_conc is None:
            return self._hold_result(channel_powers)

        self._update_count += 1

        # Update running statistics for adaptive normalization
        self._asym_mean += self._norm_alpha * (raw_asym - self._asym_mean)
        self._asym_var += self._norm_alpha * (
            (raw_asym - self._asym_mean) ** 2 - self._asym_var
        )
        self._conc_mean += self._norm_alpha * (raw_conc - self._conc_mean)
        self._conc_var += self._norm_alpha * (
            (raw_conc - self._conc_mean) ** 2 - self._conc_var
        )

        # Normalize to z-score, then squash with tanh
        asym_std = max(np.sqrt(self._asym_var), 1e-6)
        conc_std = max(np.sqrt(self._conc_var), 1e-6)

        norm_asym = float(np.tanh((raw_asym - self._asym_mean) / asym_std))
        norm_conc = float(np.tanh((raw_conc - self._conc_mean) / conc_std))
        # Map concentration from [-1,1] to [0,1]
        norm_conc = (norm_conc + 1.0) / 2.0

        # Exponential moving average smoothing
        self._smooth_asym += self._smooth_alpha * (norm_asym - self._smooth_asym)
        self._smooth_conc += self._smooth_alpha * (norm_conc - self._smooth_conc)

        self._last_asym = self._smooth_asym
        self._last_conc = self._smooth_conc

        return {
            "asymmetry": round(self._smooth_asym, 4),
            "concentration": round(self._smooth_conc, 4),
            "raw_asymmetry": round(raw_asym, 4),
            "raw_concentration": round(raw_conc, 4),
            "calibrated": self._update_count >= 10,
            "update_count": self._update_count,
            "per_channel": {
                name: {
                    "alpha": round(p["alpha"], 2),
                    "beta": round(p["beta"], 2),
                    "rejected": p["rejected"],
                    "issues": p["issues"],
                    "rms_uv": round(p["rms_uv"], 1),
                    "line_noise_db": round(p["line_noise_db"], 1),
                }
                for name, p in channel_powers.items()
            },
        }

    def _hold_result(self, channel_powers: dict) -> dict:
        """Return last good values when all channels are rejected."""
        return {
            "asymmetry": round(self._last_asym, 4),
            "concentration": round(self._last_conc, 4),
            "raw_asymmetry": 0.0,
            "raw_concentration": 0.0,
            "calibrated": self._update_count >= 10,
            "update_count": self._update_count,
            "per_channel": {
                name: {
                    "alpha": round(p["alpha"], 2),
                    "beta": round(p["beta"], 2),
                    "rejected": p["rejected"],
                    "issues": p["issues"],
                    "rms_uv": round(p["rms_uv"], 1),
                    "line_noise_db": round(p["line_noise_db"], 1),
                }
                for name, p in channel_powers.items()
            }
            if channel_powers
            else {},
        }


def _compute_channel_powers(
    data: np.ndarray,
    eeg_channels: list[int],
    channel_names: list[str],
    sampling_rate: int,
) -> dict:
    """Compute per-channel alpha and beta band powers."""
    results = {}
    for i, ch_row in enumerate(eeg_channels):
        name = channel_names[i]
        ch_data = data[ch_row, :].astype(np.float64)
        ch_data = np.nan_to_num(ch_data, nan=0.0, posinf=0.0, neginf=0.0)

        if len(ch_data) < 8:
            results[name] = {
                "alpha": 0.0,
                "beta": 0.0,
                "rejected": True,
                "issues": ["insufficient_data"],
                "rms_uv": 0.0,
                "line_noise_db": 0.0,
            }
            continue

        rms_uv = float(np.std(ch_data))
        nperseg = min(len(ch_data), sampling_rate * 2)
        freqs, psd = welch(ch_data, fs=sampling_rate, nperseg=nperseg)
        psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)

        alpha_mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
        beta_mask = (freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1])

        alpha_power = float(np.sum(psd[alpha_mask])) if np.any(alpha_mask) else 0.0
        beta_power = float(np.sum(psd[beta_mask])) if np.any(beta_mask) else 0.0

        line_mask = np.abs(freqs - LINE_NOISE_BAND_HZ) < 2.0
        broadband_mask = (freqs >= 5.0) & (freqs <= 45.0)
        line_power = float(np.mean(psd[line_mask])) if np.any(line_mask) else 0.0
        broadband_power = float(np.mean(psd[broadband_mask])) if np.any(broadband_mask) else 1e-12
        line_noise_db = float(
            10.0 * np.log10(max(line_power, 1e-12) / max(broadband_power, 1e-12))
        )

        dc_mask = freqs < 1.0
        dc_power = float(np.mean(psd[dc_mask])) if np.any(dc_mask) else 0.0
        dc_drift_uv = float(np.sqrt(max(dc_power, 0.0)))

        issues: list[str] = []
        if rms_uv < MIN_RMS_UV:
            issues.append("flat_signal")
        elif rms_uv > MAX_RMS_UV:
            issues.append("high_noise")
        if line_noise_db > MAX_LINE_NOISE_DB:
            issues.append("high_line_noise")
        if dc_drift_uv > MAX_DC_DRIFT_UV:
            issues.append("dc_drift")

        rejected = bool(issues)

        results[name] = {
            "alpha": alpha_power,
            "beta": beta_power,
            "rejected": rejected,
            "issues": issues,
            "rms_uv": rms_uv,
            "line_noise_db": line_noise_db,
        }

    return results


def _compute_asymmetry(channel_powers: dict) -> float | None:
    """Alpha asymmetry index from posterior channels. Positive = attend right."""
    left_vals = [
        channel_powers[ch]["alpha"]
        for ch in ASYMMETRY_LEFT
        if ch in channel_powers and not channel_powers[ch]["rejected"]
    ]
    right_vals = [
        channel_powers[ch]["alpha"]
        for ch in ASYMMETRY_RIGHT
        if ch in channel_powers and not channel_powers[ch]["rejected"]
    ]

    if not left_vals or not right_vals:
        return None

    left = np.mean(left_vals)
    right = np.mean(right_vals)
    denom = left + right
    if denom < 1e-12:
        return 0.0
    return float((right - left) / denom)


def _compute_concentration(channel_powers: dict) -> float | None:
    """Beta/alpha ratio from central+parietal channels."""
    alphas = []
    betas = []
    for ch in CONCENTRATION_CHANNELS:
        if ch in channel_powers and not channel_powers[ch]["rejected"]:
            alphas.append(channel_powers[ch]["alpha"])
            betas.append(channel_powers[ch]["beta"])

    if not alphas:
        return None

    alpha_mean = np.mean(alphas)
    beta_mean = np.mean(betas)
    if alpha_mean < 1e-12:
        return 0.0
    return float(beta_mean / alpha_mean)
