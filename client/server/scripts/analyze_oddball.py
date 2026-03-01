"""
Analyze an auditory oddball session (two- or three-stimulus).

Extracts epochs around each oddball marker, performs bandpass filtering,
baseline correction, artifact rejection, and computes ERPs for:
  - P3b: target vs standard (250-500ms, parietal)
  - P3a: novel vs standard (250-500ms, frontal) — if novel stimuli present
  - MMN: target vs standard (150-250ms, frontocentral)

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
ARTIFACT_THRESHOLD_UV = 150  # relaxed for dry electrodes
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 30.0
P300_WINDOW = (250, 500)  # ms post-stimulus
MMN_WINDOW = (150, 250)  # ms post-stimulus

ODDBALL_CODES = ("oddball_target", "oddball_standard", "oddball_novel")


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
    """Extract epochs around each oddball marker, grouped by condition."""
    pre_samples = int(PRE_STIMULUS_MS * sr / 1000)
    post_samples = int(POST_STIMULUS_MS * sr / 1000)
    t0 = timestamps[0]
    n_samples = eeg.shape[1]
    epoch_len = pre_samples + post_samples

    epochs: dict[str, list] = {"target": [], "standard": [], "novel": []}
    rejected = 0
    total = 0

    for m in meta["markers"]:
        if m["code"] not in ODDBALL_CODES:
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

        condition = m["code"].replace("oddball_", "")
        epochs[condition].append(epoch)

    result = {}
    for cond, ep_list in epochs.items():
        result[cond] = np.array(ep_list) if ep_list else np.empty((0, eeg.shape[0], epoch_len))

    return result, rejected, total


def compute_erp_metrics(condition_epochs, baseline_epochs, sr, ch_names, window):
    """Compute ERP peak amplitude, latency, and significance per channel."""
    pre_samples = int(PRE_STIMULUS_MS * sr / 1000)
    win_start_samp = int(window[0] * sr / 1000)
    win_end_samp = int(window[1] * sr / 1000)

    cond_erp = condition_epochs.mean(axis=0)
    base_erp = baseline_epochs.mean(axis=0)
    diff_erp = cond_erp - base_erp

    results = {}
    for ch_idx, ch_name in enumerate(ch_names):
        win_start = pre_samples + win_start_samp
        win_end = pre_samples + win_end_samp
        diff_slice = diff_erp[ch_idx, win_start:win_end]
        peak_amp = float(diff_slice.max())
        peak_latency_ms = window[0] + float(np.argmax(diff_slice)) * 1000 / sr
        neg_peak_amp = float(diff_slice.min())
        neg_peak_latency_ms = window[0] + float(np.argmin(diff_slice)) * 1000 / sr

        # t-test on mean amplitude in window
        cond_window = condition_epochs[:, ch_idx, win_start:win_end].mean(axis=1)
        base_window = baseline_epochs[:, ch_idx, win_start:win_end].mean(axis=1)

        if len(cond_window) > 1 and len(base_window) > 1:
            t_stat, p_value = ttest_ind(cond_window, base_window)
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

    return results, cond_erp, base_erp, diff_erp


def print_erp_table(title, results, ch_names, positive=True):
    """Print an ERP analysis table."""
    print("=" * 75)
    print(title)
    print("=" * 75)
    if positive:
        print(f"  {'Channel':6s}  {'Peak (µV)':>10s}  {'Latency':>8s}  {'t-stat':>8s}  {'p-value':>8s}  {'Sig':>4s}")
    else:
        print(f"  {'Channel':6s}  {'Neg Peak':>10s}  {'Latency':>8s}  {'t-stat':>8s}  {'p-value':>8s}  {'Sig':>4s}")
    print("-" * 75)

    for ch_name in ch_names:
        r = results[ch_name]
        sig = " ***" if r["significant"] else ""
        key = "peak_amplitude_uv" if positive else "neg_peak_amplitude_uv"
        lat_key = "peak_latency_ms" if positive else "neg_peak_latency_ms"
        print(
            f"  {ch_name:6s}  {r[key]:+10.2f}  {r[lat_key]:7.0f}ms"
            f"  {r['t_statistic']:+8.2f}  {r['p_value']:8.4f}  {sig}"
        )
    print()


def erp_table_md(title, results, ch_names, positive=True):
    """Generate markdown ERP table."""
    lines = [f"## {title}", ""]
    key = "peak_amplitude_uv" if positive else "neg_peak_amplitude_uv"
    lat_key = "peak_latency_ms" if positive else "neg_peak_latency_ms"
    col_name = "Peak (µV)" if positive else "Neg Peak (µV)"

    lines.append(f"| Channel | {col_name} | Latency | t-stat | p-value | Sig |")
    lines.append(f"|---------|{'---' * 5}|---------|--------|---------|-----|")

    for ch_name in ch_names:
        r = results[ch_name]
        sig = "**Yes**" if r["significant"] else "No"
        lines.append(
            f"| {ch_name} | {r[key]:+.2f} | {r[lat_key]:.0f}ms"
            f" | {r['t_statistic']:+.2f} | {r['p_value']:.4f} | {sig} |"
        )
    lines.append("")
    return lines


def generate_oddball_report(
    meta, eeg, epoch_counts, ch_names, sr,
    n_rejected, n_total,
    p3b_results, mmn_results, p3a_results=None,
) -> str:
    """Generate markdown report."""
    lines = []
    started = datetime.fromtimestamp(meta["started_at"])

    lines.append("# Auditory Oddball — Analysis Report")
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
    lines.append(f"| Target epochs | {epoch_counts['target']} |")
    lines.append(f"| Standard epochs | {epoch_counts['standard']} |")
    if epoch_counts["novel"] > 0:
        lines.append(f"| Novel epochs | {epoch_counts['novel']} |")
    lines.append(f"| Rejected (artifact/boundary) | {n_rejected} |")
    lines.append(f"| Rejection rate | {rejection_rate:.1f}% |")
    lines.append("")

    # P3b: target vs standard
    lines.extend(erp_table_md("P3b — Target vs Standard (250–500 ms)", p3b_results, ch_names))

    # P3a: novel vs standard (if present)
    if p3a_results:
        lines.extend(erp_table_md("P3a — Novel vs Standard (250–500 ms)", p3a_results, ch_names))

    # MMN: target vs standard
    lines.extend(erp_table_md("MMN — Target vs Standard (150–250 ms)", mmn_results, ch_names, positive=False))

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
    epochs, n_rejected, n_total = extract_oddball_epochs(meta, eeg_filt, timestamps, sr)
    epoch_counts = {k: len(v) for k, v in epochs.items()}

    print(f"Epochs: {epoch_counts['target']} target, {epoch_counts['standard']} standard, "
          f"{epoch_counts['novel']} novel, {n_rejected} rejected / {n_total} total")
    print()

    if epoch_counts["target"] < 2 or epoch_counts["standard"] < 2:
        print("Not enough epochs for ERP analysis.")
        return

    # P3b: target vs standard (250-500 ms)
    p3b_results, _, _, _ = compute_erp_metrics(
        epochs["target"], epochs["standard"], sr, ch_names, P300_WINDOW
    )

    # MMN: target vs standard (150-250 ms)
    mmn_results, _, _, _ = compute_erp_metrics(
        epochs["target"], epochs["standard"], sr, ch_names, MMN_WINDOW
    )

    # P3a: novel vs standard (if novels present)
    p3a_results = None
    if epoch_counts["novel"] >= 2:
        p3a_results, _, _, _ = compute_erp_metrics(
            epochs["novel"], epochs["standard"], sr, ch_names, P300_WINDOW
        )

    if args.report:
        report = generate_oddball_report(
            meta, eeg, epoch_counts, ch_names, sr,
            n_rejected, n_total,
            p3b_results, mmn_results, p3a_results,
        )
        session_dir = SESSIONS_DIR / meta["session_id"]
        report_path = session_dir / "report.md"
        report_path.write_text(report)
        print(f"Report written to {report_path}")
    else:
        print_erp_table("P3b — TARGET vs STANDARD (250-500 ms)", p3b_results, ch_names)
        if p3a_results:
            print_erp_table("P3a — NOVEL vs STANDARD (250-500 ms)", p3a_results, ch_names)
        print_erp_table("MMN — TARGET vs STANDARD (150-250 ms, negative peak)", mmn_results, ch_names, positive=False)


if __name__ == "__main__":
    main()
