"""
Analyze a recorded experiment session.

Loads raw EEG + markers, computes per-block band power using Welch's method,
and prints a comparison across conditions (e.g., eyes open vs closed).

Usage:
    cd server/
    uv run python scripts/analyze_session.py                     # latest session, print
    uv run python scripts/analyze_session.py <session_id>        # specific session
    uv run python scripts/analyze_session.py --report            # latest session, write report.md
    uv run python scripts/analyze_session.py --report <id>       # specific session, write report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import welch

SESSIONS_DIR = Path(__file__).resolve().parent.parent.parent / "sessions"

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}


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


def extract_blocks(meta, timestamps, sr, n_samples):
    """Convert markers into block sample ranges."""
    blocks = []
    current = None
    for m in meta["markers"]:
        if m["code"] == "block_start":
            current = {"block_id": m["block_id"], "start_ts": m["server_timestamp"]}
        elif m["code"] == "block_end" and current:
            current["end_ts"] = m["server_timestamp"]
            blocks.append(current)
            current = None

    t0 = timestamps[0]
    for b in blocks:
        b["start_idx"] = max(0, min(int((b["start_ts"] - t0) * sr), n_samples))
        b["end_idx"] = max(0, min(int((b["end_ts"] - t0) * sr), n_samples))
        b["n_samples"] = b["end_idx"] - b["start_idx"]
    return blocks


def compute_block_bandpower(eeg, blocks, sr, ch_names):
    """Compute Welch PSD and band powers for each block."""
    results = {}
    for b in blocks:
        epoch = eeg[:, b["start_idx"] : b["end_idx"]]
        if epoch.shape[1] < sr:
            continue

        f, pxx = welch(epoch, fs=sr, nperseg=min(512, epoch.shape[1]), noverlap=256)

        powers = {}
        for band_name, (lo, hi) in BANDS.items():
            mask = (f >= lo) & (f <= hi)
            powers[band_name] = pxx[:, mask].mean(axis=1)

        total_mask = (f >= 0.5) & (f <= 50)
        total = pxx[:, total_mask].mean(axis=1)

        results[b["block_id"]] = {
            "powers": powers,
            "total": total,
            "psd": pxx,
            "freqs": f,
        }
    return results


def print_analysis(results, ch_names):
    """Print formatted analysis results."""
    # --- Alpha power per channel ---
    print("=" * 60)
    print("ALPHA POWER ANALYSIS (8-13 Hz)")
    print("=" * 60)

    for ch_idx, ch_name in enumerate(ch_names):
        open_alphas = []
        closed_alphas = []

        for block_id, r in results.items():
            alpha = r["powers"]["alpha"][ch_idx]
            total = r["total"][ch_idx]
            relative = alpha / total if total > 0 else 0

            if "open" in block_id:
                open_alphas.append(relative)
            else:
                closed_alphas.append(relative)

        mean_open = np.mean(open_alphas) if open_alphas else 0
        mean_closed = np.mean(closed_alphas) if closed_alphas else 0
        ratio = mean_closed / mean_open if mean_open > 0 else 0

        marker = "***" if ratio > 1.2 else "  *" if ratio > 1.0 else "   "
        print(
            f"  {ch_name:4s}  open={mean_open:.3f}  closed={mean_closed:.3f}"
            f"  ratio={ratio:.2f}x  {marker}"
        )

    # --- Full band power by condition ---
    print()
    print("=" * 60)
    print("FULL BAND POWER BY CONDITION (relative)")
    print("=" * 60)

    for condition in ["open", "closed"]:
        cond_blocks = {k: v for k, v in results.items() if condition in k}
        if not cond_blocks:
            continue

        print(f"\n  Eyes {condition.upper()}:")
        for band_name in ["delta", "theta", "alpha", "beta", "gamma"]:
            vals = []
            for r in cond_blocks.values():
                rel = r["powers"][band_name] / r["total"]
                vals.append(rel)
            avg = np.mean(vals, axis=0)
            ch_str = "  ".join(f"{ch_names[i]}={avg[i]:.3f}" for i in range(len(ch_names)))
            print(f"    {band_name:6s} (avg={np.mean(avg):.3f}): {ch_str}")

    # --- Occipital focus ---
    if "O1" in ch_names and "O2" in ch_names:
        o1 = ch_names.index("O1")
        o2 = ch_names.index("O2")

        print()
        print("=" * 60)
        print("OCCIPITAL FOCUS (O1, O2) — key channels for alpha")
        print("=" * 60)

        for band_name in ["delta", "theta", "alpha", "beta", "gamma"]:
            open_vals = []
            closed_vals = []
            for block_id, r in results.items():
                rel = r["powers"][band_name] / r["total"]
                occ_mean = (rel[o1] + rel[o2]) / 2
                if "open" in block_id:
                    open_vals.append(occ_mean)
                else:
                    closed_vals.append(occ_mean)

            mo = np.mean(open_vals)
            mc = np.mean(closed_vals)
            ratio = mc / mo if mo > 0 else 0
            print(f"  {band_name:6s}  open={mo:.3f}  closed={mc:.3f}  ratio={ratio:.2f}x")


def generate_report(meta, eeg, blocks, results, sr, ch_names) -> str:
    """Generate a markdown report from analysis results."""
    lines = []
    started = datetime.fromtimestamp(meta["started_at"])

    lines.append(f"# {meta['protocol_id']} — Analysis Report")
    lines.append("")
    lines.append(f"**Session:** `{meta['session_id']}`  ")
    lines.append(f"**Date:** {started.strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Duration:** {meta['duration_sec']:.1f}s  ")
    lines.append(f"**Samples:** {eeg.shape[1]:,} ({eeg.shape[0]} channels @ {sr} Hz)")
    lines.append("")

    # Blocks summary
    lines.append("## Blocks")
    lines.append("")
    lines.append("| Block | Samples | Duration |")
    lines.append("|-------|---------|----------|")
    for b in blocks:
        lines.append(f"| {b['block_id']} | {b['n_samples']:,} | {b['n_samples']/sr:.1f}s |")
    lines.append("")

    # Alpha analysis
    lines.append("## Alpha Power Analysis (8–13 Hz)")
    lines.append("")
    lines.append("| Channel | Eyes Open | Eyes Closed | Ratio |")
    lines.append("|---------|-----------|-------------|-------|")

    for ch_idx, ch_name in enumerate(ch_names):
        open_alphas = []
        closed_alphas = []
        for block_id, r in results.items():
            alpha = r["powers"]["alpha"][ch_idx]
            total = r["total"][ch_idx]
            relative = alpha / total if total > 0 else 0
            if "open" in block_id:
                open_alphas.append(relative)
            else:
                closed_alphas.append(relative)

        mean_open = np.mean(open_alphas) if open_alphas else 0
        mean_closed = np.mean(closed_alphas) if closed_alphas else 0
        ratio = mean_closed / mean_open if mean_open > 0 else 0
        marker = " **" if ratio > 1.2 else ""
        lines.append(f"| {ch_name} | {mean_open:.3f} | {mean_closed:.3f} | {ratio:.2f}x{marker} |")

    lines.append("")

    # Full band power
    lines.append("## Band Power by Condition")
    lines.append("")
    for condition in ["open", "closed"]:
        cond_blocks = {k: v for k, v in results.items() if condition in k}
        if not cond_blocks:
            continue
        lines.append(f"### Eyes {condition.capitalize()}")
        lines.append("")
        header = "| Band | " + " | ".join(ch_names) + " | Avg |"
        sep = "|------|" + "|".join(["------"] * len(ch_names)) + "|-----|"
        lines.append(header)
        lines.append(sep)
        for band_name in ["delta", "theta", "alpha", "beta", "gamma"]:
            vals = []
            for r in cond_blocks.values():
                rel = r["powers"][band_name] / r["total"]
                vals.append(rel)
            avg = np.mean(vals, axis=0)
            row = f"| {band_name} | " + " | ".join(f"{avg[i]:.3f}" for i in range(len(ch_names)))
            row += f" | {np.mean(avg):.3f} |"
            lines.append(row)
        lines.append("")

    # Occipital focus
    if "O1" in ch_names and "O2" in ch_names:
        o1 = ch_names.index("O1")
        o2 = ch_names.index("O2")
        lines.append("## Occipital Focus (O1, O2)")
        lines.append("")
        lines.append("Key channels for alpha rhythm detection.")
        lines.append("")
        lines.append("| Band | Eyes Open | Eyes Closed | Ratio |")
        lines.append("|------|-----------|-------------|-------|")
        for band_name in ["delta", "theta", "alpha", "beta", "gamma"]:
            open_vals = []
            closed_vals = []
            for block_id, r in results.items():
                rel = r["powers"][band_name] / r["total"]
                occ_mean = (rel[o1] + rel[o2]) / 2
                if "open" in block_id:
                    open_vals.append(occ_mean)
                else:
                    closed_vals.append(occ_mean)
            mo = np.mean(open_vals)
            mc = np.mean(closed_vals)
            ratio = mc / mo if mo > 0 else 0
            lines.append(f"| {band_name} | {mo:.3f} | {mc:.3f} | {ratio:.2f}x |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze a recorded EEG session")
    parser.add_argument("session_id", nargs="?", default=None, help="Session ID (default: latest)")
    parser.add_argument("--report", action="store_true", help="Write report.md to session directory")
    args = parser.parse_args()

    meta, eeg, timestamps, sr, ch_names = load_session(args.session_id)

    print(f"Session: {meta['session_id']}")
    print(f"Protocol: {meta['protocol_id']}")
    print(f"Duration: {meta['duration_sec']:.1f}s")
    print(f"Samples: {eeg.shape[1]} ({eeg.shape[0]} channels @ {sr} Hz)")
    print()

    blocks = extract_blocks(meta, timestamps, sr, eeg.shape[1])
    for b in blocks:
        print(f"  {b['block_id']:20s}  {b['n_samples']:5d} samples  ({b['n_samples']/sr:.1f}s)")
    print()

    results = compute_block_bandpower(eeg, blocks, sr, ch_names)

    if args.report:
        report = generate_report(meta, eeg, blocks, results, sr, ch_names)
        session_dir = SESSIONS_DIR / meta["session_id"]
        report_path = session_dir / "report.md"
        report_path.write_text(report)
        print(f"Report written to {report_path}")
    else:
        print_analysis(results, ch_names)


if __name__ == "__main__":
    main()
