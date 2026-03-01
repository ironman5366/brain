"""
Analyze a P300 speller session.

Extracts epochs around each flash marker, performs baseline correction,
averages by condition (target vs non-target), and detects P300 component.

Usage:
    cd server/
    uv run python scripts/analyze_p300.py                  # latest session
    uv run python scripts/analyze_p300.py <session_id>     # specific session
    uv run python scripts/analyze_p300.py --report         # write report.md
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
ARTIFACT_THRESHOLD_UV = 100
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 30.0
P300_WINDOW = (250, 500)  # ms post-stimulus


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


def extract_flash_epochs(meta, eeg, timestamps, sr):
    """Extract epochs around each p300_flash marker."""
    pre_samples = int(PRE_STIMULUS_MS * sr / 1000)
    post_samples = int(POST_STIMULUS_MS * sr / 1000)
    t0 = timestamps[0]
    n_samples = eeg.shape[1]

    target_epochs = []
    nontarget_epochs = []
    rejected = 0
    total = 0

    for m in meta["markers"]:
        if m["code"] != "p300_flash":
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

        is_target = m["metadata"]["is_target"]
        if is_target:
            target_epochs.append(epoch)
        else:
            nontarget_epochs.append(epoch)

    target_arr = np.array(target_epochs) if target_epochs else np.empty((0, eeg.shape[0], pre_samples + post_samples))
    nontarget_arr = np.array(nontarget_epochs) if nontarget_epochs else np.empty((0, eeg.shape[0], pre_samples + post_samples))

    return target_arr, nontarget_arr, rejected, total


def extract_per_character(meta, eeg, timestamps, sr):
    """Extract epochs grouped by target character."""
    pre_samples = int(PRE_STIMULUS_MS * sr / 1000)
    post_samples = int(POST_STIMULUS_MS * sr / 1000)
    t0 = timestamps[0]
    n_samples = eeg.shape[1]

    chars: dict[str, dict] = {}

    for m in meta["markers"]:
        if m["code"] != "p300_flash":
            continue

        md = m["metadata"]
        letter = md["target_letter"]
        if letter not in chars:
            chars[letter] = {"target": [], "nontarget": []}

        sample_idx = int((m["server_timestamp"] - t0) * sr)
        start = sample_idx - pre_samples
        end = sample_idx + post_samples

        if start < 0 or end > n_samples:
            continue

        epoch = eeg[:, start:end].copy()
        baseline = epoch[:, :pre_samples].mean(axis=1, keepdims=True)
        epoch -= baseline

        if np.any(np.abs(epoch) > ARTIFACT_THRESHOLD_UV):
            continue

        if md["is_target"]:
            chars[letter]["target"].append(epoch)
        else:
            chars[letter]["nontarget"].append(epoch)

    return chars


def compute_erp_metrics(target_epochs, nontarget_epochs, sr, ch_names):
    """Compute P300 peak amplitude, latency, and significance per channel."""
    pre_samples = int(PRE_STIMULUS_MS * sr / 1000)
    p300_start_samp = int(P300_WINDOW[0] * sr / 1000)
    p300_end_samp = int(P300_WINDOW[1] * sr / 1000)

    target_erp = target_epochs.mean(axis=0)
    nontarget_erp = nontarget_epochs.mean(axis=0)
    diff_erp = target_erp - nontarget_erp

    n_time = target_erp.shape[1]
    time_ms = np.arange(n_time) * 1000 / sr - PRE_STIMULUS_MS

    results = {}
    for ch_idx, ch_name in enumerate(ch_names):
        # P300 window in the diff waveform
        win_start = pre_samples + p300_start_samp
        win_end = pre_samples + p300_end_samp
        p300_slice = diff_erp[ch_idx, win_start:win_end]
        peak_amp = float(p300_slice.max())
        peak_latency_ms = P300_WINDOW[0] + float(np.argmax(p300_slice)) * 1000 / sr

        # t-test on mean amplitude in P300 window
        target_window = target_epochs[:, ch_idx, win_start:win_end].mean(axis=1)
        nontarget_window = nontarget_epochs[:, ch_idx, win_start:win_end].mean(axis=1)

        if len(target_window) > 1 and len(nontarget_window) > 1:
            t_stat, p_value = ttest_ind(target_window, nontarget_window)
        else:
            t_stat, p_value = 0.0, 1.0

        results[ch_name] = {
            "peak_amplitude_uv": peak_amp,
            "peak_latency_ms": peak_latency_ms,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    return results, target_erp, nontarget_erp, diff_erp, time_ms


def print_p300_analysis(erp_results, ch_names, n_target, n_nontarget, n_rejected, n_total):
    """Print P300 analysis to stdout."""
    print(f"Epochs: {n_target} target, {n_nontarget} non-target, {n_rejected} rejected / {n_total} total")
    print()

    print("=" * 70)
    print("P300 ERP ANALYSIS (250-500 ms window)")
    print("=" * 70)
    print(f"  {'Channel':6s}  {'Peak (µV)':>10s}  {'Latency':>8s}  {'t-stat':>8s}  {'p-value':>8s}  {'Sig':>4s}")
    print("-" * 70)

    for ch_name in ch_names:
        r = erp_results[ch_name]
        sig = " ***" if r["significant"] else ""
        print(
            f"  {ch_name:6s}  {r['peak_amplitude_uv']:+10.2f}  {r['peak_latency_ms']:7.0f}ms"
            f"  {r['t_statistic']:+8.2f}  {r['p_value']:8.4f}  {sig}"
        )
    print()


def generate_p300_report(meta, eeg, erp_results, ch_names, sr, n_target, n_nontarget, n_rejected, n_total, per_char) -> str:
    """Generate markdown report."""
    lines = []
    started = datetime.fromtimestamp(meta["started_at"])

    lines.append(f"# {meta['protocol_id']} — Analysis Report")
    lines.append("")
    lines.append(f"**Session:** `{meta['session_id']}`  ")
    lines.append(f"**Date:** {started.strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Duration:** {meta['duration_sec']:.1f}s  ")
    lines.append(f"**Samples:** {eeg.shape[1]:,} ({eeg.shape[0]} channels @ {sr} Hz)")
    lines.append("")

    # Epoch stats
    lines.append("## Epoch Summary")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total flash events | {n_total} |")
    lines.append(f"| Target epochs | {n_target} |")
    lines.append(f"| Non-target epochs | {n_nontarget} |")
    lines.append(f"| Rejected (artifact/boundary) | {n_rejected} |")
    lines.append(f"| Rejection rate | {n_rejected/n_total*100:.1f}% |" if n_total > 0 else "| Rejection rate | N/A |")
    lines.append("")

    # ERP metrics
    lines.append("## P300 ERP Analysis (250–500 ms window)")
    lines.append("")
    lines.append("| Channel | Peak (µV) | Latency | t-stat | p-value | Sig |")
    lines.append("|---------|-----------|---------|--------|---------|-----|")

    for ch_name in ch_names:
        r = erp_results[ch_name]
        sig = "**Yes**" if r["significant"] else "No"
        lines.append(
            f"| {ch_name} | {r['peak_amplitude_uv']:+.2f} | {r['peak_latency_ms']:.0f}ms"
            f" | {r['t_statistic']:+.2f} | {r['p_value']:.4f} | {sig} |"
        )
    lines.append("")

    # Per-character breakdown
    if per_char:
        lines.append("## Per-Character Breakdown")
        lines.append("")
        for letter, epochs in per_char.items():
            n_t = len(epochs["target"])
            n_nt = len(epochs["nontarget"])
            lines.append(f"### Target: {letter}")
            lines.append(f"")
            lines.append(f"{n_t} target epochs, {n_nt} non-target epochs")
            lines.append("")

            if n_t >= 2 and n_nt >= 2:
                t_arr = np.array(epochs["target"])
                nt_arr = np.array(epochs["nontarget"])
                char_results, _, _, _, _ = compute_erp_metrics(t_arr, nt_arr, sr, ch_names)

                # Show just P7/P8 (parietal — best for P300)
                lines.append("| Channel | Peak (µV) | Latency | p-value |")
                lines.append("|---------|-----------|---------|---------|")
                for ch in ch_names:
                    cr = char_results[ch]
                    lines.append(f"| {ch} | {cr['peak_amplitude_uv']:+.2f} | {cr['peak_latency_ms']:.0f}ms | {cr['p_value']:.4f} |")
                lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze a P300 speller session")
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
    target_epochs, nontarget_epochs, n_rejected, n_total = extract_flash_epochs(
        meta, eeg_filt, timestamps, sr
    )
    n_target = len(target_epochs)
    n_nontarget = len(nontarget_epochs)

    print(f"Epochs: {n_target} target, {n_nontarget} non-target, {n_rejected} rejected / {n_total} total")
    print()

    if n_target < 2 or n_nontarget < 2:
        print("Not enough epochs for ERP analysis.")
        return

    erp_results, _, _, _, _ = compute_erp_metrics(target_epochs, nontarget_epochs, sr, ch_names)

    if args.report:
        per_char = extract_per_character(meta, eeg_filt, timestamps, sr)
        report = generate_p300_report(
            meta, eeg, erp_results, ch_names, sr,
            n_target, n_nontarget, n_rejected, n_total, per_char,
        )
        session_dir = SESSIONS_DIR / meta["session_id"]
        report_path = session_dir / "report.md"
        report_path.write_text(report)
        print(f"Report written to {report_path}")
    else:
        print_p300_analysis(erp_results, ch_names, n_target, n_nontarget, n_rejected, n_total)


if __name__ == "__main__":
    main()
