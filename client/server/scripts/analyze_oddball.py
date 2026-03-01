"""
Analyze an auditory oddball P300 session.

Extracts epochs around each oddball_target and oddball_standard marker,
performs bandpass filtering, baseline correction, artifact rejection,
and computes ERPs for P300 (250-500ms) and MMN (150-250ms) components.

Usage:
    cd server/
    uv run python scripts/analyze_oddball.py                  # latest session
    uv run python scripts/analyze_oddball.py <session_id>     # specific session
    uv run python scripts/analyze_oddball.py --report         # write report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import ttest_ind

SESSIONS_DIR = Path(__file__).resolve().parent.parent.parent / "sessions"

# ERP parameters
PRE_STIMULUS_MS = 100
POST_STIMULUS_MS = 600
ARTIFACT_THRESHOLD_UV = 150  # relaxed vs P300 (dry electrodes)
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 30.0
P300_WINDOW = (250, 500)  # ms post-stimulus
MMN_WINDOW = (150, 250)  # ms post-stimulus


def load_session(session_id: str | None = None):
    """Load a session by ID, or the most recent one."""
    if session_id:
        session_dir = SESSIONS_DIR / session_id
    else:
        dirs = sorted(SESSIONS_DIR.iterdir(), reverse=True)
        if not dirs:
            print("No sessions found.")
            sys.exit(1)
        session_dir = dirs[0]

    with open(session_dir / "session.json") as f:
        meta = json.load(f)

    data = np.load(session_dir / "eeg_raw.npz")
    eeg = data["eeg"]
    timestamps = data["timestamps"]
    sr = int(data["sampling_rate"])
    ch_names = list(data["channel_names"])

    return meta, eeg, timestamps, sr, ch_names


def bandpass_filter(eeg, sr, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=4):
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = sr / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, eeg, axis=1)


def extract_oddball_epochs(meta, eeg, timestamps, sr):
    """Extract epochs around each oddball marker."""
    pre_samples = int(PRE_STIMULUS_MS * sr / 1000)
    post_samples = int(POST_STIMULUS_MS * sr / 1000)
    t0 = timestamps[0]
    n_samples = eeg.shape[1]

    target_epochs = []
    standard_epochs = []
    rejected = 0
    total = 0

    for m in meta["markers"]:
        if m["code"] not in ("oddball_target", "oddball_standard"):
            continue

        total += 1
        sample_idx = int((m["server_timestamp"] - t0) * sr)
        start = sample_idx - pre_samples
        end = sample_idx + post_samples

        if start < 0 or end > n_samples:
            rejected += 1
            continue

        epoch = eeg[:, start:end].copy()

        # Baseline correction
        baseline = epoch[:, :pre_samples].mean(axis=1, keepdims=True)
        epoch -= baseline

        # Artifact rejection
        if np.any(np.abs(epoch) > ARTIFACT_THRESHOLD_UV):
            rejected += 1
            continue

        if m["code"] == "oddball_target":
            target_epochs.append(epoch)
        else:
            standard_epochs.append(epoch)

    epoch_len = pre_samples + post_samples
    target_arr = np.array(target_epochs) if target_epochs else np.empty((0, eeg.shape[0], epoch_len))
    standard_arr = np.array(standard_epochs) if standard_epochs else np.empty((0, eeg.shape[0], epoch_len))

    return target_arr, standard_arr, rejected, total


def compute_erp_metrics(target_epochs, standard_epochs, sr, ch_names, window):
    """Compute ERP peak amplitude, latency, and significance per channel in the given window."""
    pre_samples = int(PRE_STIMULUS_MS * sr / 1000)
    win_start_samp = int(window[0] * sr / 1000)
    win_end_samp = int(window[1] * sr / 1000)

    target_erp = target_epochs.mean(axis=0)
    standard_erp = standard_epochs.mean(axis=0)
    diff_erp = target_erp - standard_erp

    results = {}
    for ch_idx, ch_name in enumerate(ch_names):
        win_start = pre_samples + win_start_samp
        win_end = pre_samples + win_end_samp
        diff_slice = diff_erp[ch_idx, win_start:win_end]
        peak_amp = float(diff_slice.max())
        peak_latency_ms = window[0] + float(np.argmax(diff_slice)) * 1000 / sr

        # Also check for negative peak (for MMN)
        neg_peak_amp = float(diff_slice.min())
        neg_peak_latency_ms = window[0] + float(np.argmin(diff_slice)) * 1000 / sr

        # t-test on mean amplitude in window
        target_window = target_epochs[:, ch_idx, win_start:win_end].mean(axis=1)
        standard_window = standard_epochs[:, ch_idx, win_start:win_end].mean(axis=1)

        if len(target_window) > 1 and len(standard_window) > 1:
            t_stat, p_value = ttest_ind(target_window, standard_window)
        else:
            t_stat, p_value = 0.0, 1.0

        results[ch_name] = {
            "peak_amplitude_uv": peak_amp,
            "peak_latency_ms": peak_latency_ms,
            "neg_peak_amplitude_uv": neg_peak_amp,
            "neg_peak_latency_ms": neg_peak_latency_ms,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    return results, target_erp, standard_erp, diff_erp


def print_oddball_analysis(p300_results, mmn_results, ch_names, n_target, n_standard, n_rejected, n_total):
    """Print analysis to stdout."""
    print(f"Epochs: {n_target} target, {n_standard} standard, {n_rejected} rejected / {n_total} total")
    print()

    print("=" * 75)
    print("P300 ANALYSIS (250-500 ms window)")
    print("=" * 75)
    print(f"  {'Channel':6s}  {'Peak (µV)':>10s}  {'Latency':>8s}  {'t-stat':>8s}  {'p-value':>8s}  {'Sig':>4s}")
    print("-" * 75)

    for ch_name in ch_names:
        r = p300_results[ch_name]
        sig = " ***" if r["significant"] else ""
        print(
            f"  {ch_name:6s}  {r['peak_amplitude_uv']:+10.2f}  {r['peak_latency_ms']:7.0f}ms"
            f"  {r['t_statistic']:+8.2f}  {r['p_value']:8.4f}  {sig}"
        )
    print()

    print("=" * 75)
    print("MMN ANALYSIS (150-250 ms window, negative peak)")
    print("=" * 75)
    print(f"  {'Channel':6s}  {'Neg Peak':>10s}  {'Latency':>8s}  {'t-stat':>8s}  {'p-value':>8s}  {'Sig':>4s}")
    print("-" * 75)

    for ch_name in ch_names:
        r = mmn_results[ch_name]
        sig = " ***" if r["significant"] else ""
        print(
            f"  {ch_name:6s}  {r['neg_peak_amplitude_uv']:+10.2f}  {r['neg_peak_latency_ms']:7.0f}ms"
            f"  {r['t_statistic']:+8.2f}  {r['p_value']:8.4f}  {sig}"
        )
    print()


def generate_oddball_report(
    meta, eeg, p300_results, mmn_results, ch_names, sr,
    n_target, n_standard, n_rejected, n_total,
    target_erp, standard_erp,
) -> str:
    """Generate markdown report."""
    lines = []
    started = datetime.fromtimestamp(meta["started_at"])

    lines.append(f"# Auditory Oddball — Analysis Report")
    lines.append("")
    lines.append(f"**Session:** `{meta['session_id']}`  ")
    lines.append(f"**Date:** {started.strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Protocol:** `{meta['protocol_id']}`  ")
    lines.append(f"**Duration:** {meta['duration_sec']:.1f}s  ")
    lines.append(f"**Recording:** {eeg.shape[1]:,} samples, {eeg.shape[0]} channels @ {sr} Hz")
    lines.append("")

    # Epoch summary
    rejection_rate = n_rejected / n_total * 100 if n_total > 0 else 0
    lines.append("## Epoch Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Total markers | {n_total} |")
    lines.append(f"| Target epochs | {n_target} |")
    lines.append(f"| Standard epochs | {n_standard} |")
    lines.append(f"| Rejected (artifact/boundary) | {n_rejected} |")
    lines.append(f"| Rejection rate | {rejection_rate:.1f}% |")
    lines.append("")

    # P300 analysis
    lines.append("## P300 Analysis (250-500 ms window)")
    lines.append("")
    lines.append("| Channel | Peak (µV) | Latency | t-stat | p-value | Sig |")
    lines.append("|---------|-----------|---------|--------|---------|-----|")

    for ch_name in ch_names:
        r = p300_results[ch_name]
        sig = "**Yes**" if r["significant"] else "No"
        lines.append(
            f"| {ch_name} | {r['peak_amplitude_uv']:+.2f} | {r['peak_latency_ms']:.0f}ms"
            f" | {r['t_statistic']:+.2f} | {r['p_value']:.4f} | {sig} |"
        )
    lines.append("")

    # MMN analysis
    lines.append("## MMN Analysis (150-250 ms window)")
    lines.append("")
    lines.append("| Channel | Neg Peak (µV) | Latency | t-stat | p-value | Sig |")
    lines.append("|---------|---------------|---------|--------|---------|-----|")

    for ch_name in ch_names:
        r = mmn_results[ch_name]
        sig = "**Yes**" if r["significant"] else "No"
        lines.append(
            f"| {ch_name} | {r['neg_peak_amplitude_uv']:+.2f} | {r['neg_peak_latency_ms']:.0f}ms"
            f" | {r['t_statistic']:+.2f} | {r['p_value']:.4f} | {sig} |"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze an auditory oddball session")
    parser.add_argument("session_id", nargs="?", default=None, help="Session ID (default: latest)")
    parser.add_argument("--report", action="store_true", help="Write report.md to session directory")
    args = parser.parse_args()

    meta, eeg, timestamps, sr, ch_names = load_session(args.session_id)

    print(f"Session: {meta['session_id']}")
    print(f"Protocol: {meta['protocol_id']}")
    print(f"Duration: {meta['duration_sec']:.1f}s")
    print(f"Samples: {eeg.shape[1]} ({eeg.shape[0]} channels @ {sr} Hz)")
    print()

    # Bandpass filter
    print("Applying bandpass filter (0.5-30 Hz)...")
    eeg_filt = bandpass_filter(eeg, sr)

    # Extract epochs
    target_epochs, standard_epochs, n_rejected, n_total = extract_oddball_epochs(
        meta, eeg_filt, timestamps, sr
    )
    n_target = len(target_epochs)
    n_standard = len(standard_epochs)

    print(f"Epochs: {n_target} target, {n_standard} standard, {n_rejected} rejected / {n_total} total")
    print()

    if n_target < 2 or n_standard < 2:
        print("Not enough epochs for ERP analysis.")
        return

    # P300 analysis (250-500 ms)
    p300_results, target_erp, standard_erp, _ = compute_erp_metrics(
        target_epochs, standard_epochs, sr, ch_names, P300_WINDOW
    )

    # MMN analysis (150-250 ms)
    mmn_results, _, _, _ = compute_erp_metrics(
        target_epochs, standard_epochs, sr, ch_names, MMN_WINDOW
    )

    if args.report:
        report = generate_oddball_report(
            meta, eeg, p300_results, mmn_results, ch_names, sr,
            n_target, n_standard, n_rejected, n_total,
            target_erp, standard_erp,
        )
        session_dir = SESSIONS_DIR / meta["session_id"]
        report_path = session_dir / "report.md"
        report_path.write_text(report)
        print(f"Report written to {report_path}")
    else:
        print_oddball_analysis(
            p300_results, mmn_results, ch_names,
            n_target, n_standard, n_rejected, n_total,
        )


if __name__ == "__main__":
    main()
