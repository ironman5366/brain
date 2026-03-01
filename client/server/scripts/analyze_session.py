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


## ── SSVEP-specific analysis ─────────────────────────────────


def extract_ssvep_info(meta):
    """Extract SSVEP stimulation parameters from markers."""
    ssvep_blocks = {}
    for m in meta["markers"]:
        if m["code"] == "ssvep_start" and m.get("metadata"):
            ssvep_blocks[m["block_id"]] = {
                "frequencies": m["metadata"]["frequencies"],
                "target_frequency": m["metadata"].get("target_frequency"),
                "duration_ms": m["metadata"].get("duration_ms"),
            }
    return ssvep_blocks


def compute_ssvep_snr(freqs, psd_row, target_hz, noise_bw=2.0):
    """
    SNR at a target frequency for a single channel's PSD.

    SNR = power at target bin / mean power in surrounding noise band.
    noise_bw: Hz on each side of target used as noise estimate,
    excluding target ± 1 bin.
    """
    freq_res = freqs[1] - freqs[0]
    target_idx = np.argmin(np.abs(freqs - target_hz))
    target_power = psd_row[target_idx]

    noise_mask = (
        (freqs >= target_hz - noise_bw)
        & (freqs <= target_hz + noise_bw)
        & (np.abs(freqs - target_hz) > freq_res * 1.5)
    )
    noise_power = np.mean(psd_row[noise_mask]) if noise_mask.any() else 1e-12

    snr = target_power / noise_power if noise_power > 0 else 0
    snr_db = 10 * np.log10(snr) if snr > 0 else float("-inf")
    return snr, snr_db, target_power, noise_power


def print_ssvep_analysis(results, ssvep_info, ch_names):
    """Print SSVEP analysis to stdout."""
    # Group blocks by type
    baseline_ids = [bid for bid in results if "baseline" in bid]
    for target_hz in sorted({info["target_frequency"] for info in ssvep_info.values() if info["target_frequency"]}):
        stim_ids = [bid for bid, info in ssvep_info.items() if info["target_frequency"] == target_hz and bid in results]

        print("=" * 60)
        print(f"SSVEP {target_hz} Hz ANALYSIS")
        print("=" * 60)

        for ch_idx, ch_name in enumerate(ch_names):
            # Average SNR across stimulation blocks
            snrs = []
            for bid in stim_ids:
                r = results[bid]
                snr, snr_db, _, _ = compute_ssvep_snr(r["freqs"], r["psd"][ch_idx], target_hz)
                snrs.append(snr_db)

            # Average SNR during baselines (should be low)
            base_snrs = []
            for bid in baseline_ids:
                if bid in results:
                    r = results[bid]
                    _, snr_db, _, _ = compute_ssvep_snr(r["freqs"], r["psd"][ch_idx], target_hz)
                    base_snrs.append(snr_db)

            stim_snr = np.mean(snrs) if snrs else 0
            base_snr = np.mean(base_snrs) if base_snrs else 0
            marker = "***" if stim_snr > 6 else "  *" if stim_snr > 3 else "   "
            print(f"  {ch_name:4s}  stim={stim_snr:+.1f} dB  baseline={base_snr:+.1f} dB  {marker}")

        # Check harmonics
        print(f"\n  Harmonics (O1/O2 avg):")
        if "O1" in ch_names and "O2" in ch_names:
            o1, o2 = ch_names.index("O1"), ch_names.index("O2")
            for harmonic in [1, 2, 3]:
                freq_check = target_hz * harmonic
                if freq_check > 50:
                    break
                snrs = []
                for bid in stim_ids:
                    r = results[bid]
                    for ci in [o1, o2]:
                        _, snr_db, _, _ = compute_ssvep_snr(r["freqs"], r["psd"][ci], freq_check)
                        snrs.append(snr_db)
                avg_snr = np.mean(snrs) if snrs else 0
                label = f"{harmonic}f = {freq_check} Hz"
                print(f"    {label:15s}  SNR={avg_snr:+.1f} dB")
        print()


def generate_ssvep_report(meta, eeg, blocks, results, ssvep_info, sr, ch_names) -> str:
    """Generate a markdown report for an SSVEP session."""
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
    lines.append("| Block | Type | Samples | Duration |")
    lines.append("|-------|------|---------|----------|")
    for b in blocks:
        bid = b["block_id"]
        btype = "SSVEP" if bid in ssvep_info else "Baseline"
        freq_str = ""
        if bid in ssvep_info:
            freq_str = f" ({ssvep_info[bid]['target_frequency']} Hz)"
        lines.append(f"| {bid} | {btype}{freq_str} | {b['n_samples']:,} | {b['n_samples']/sr:.1f}s |")
    lines.append("")

    # SNR analysis per frequency
    baseline_ids = [bid for bid in results if "baseline" in bid]
    target_freqs = sorted({info["target_frequency"] for info in ssvep_info.values() if info["target_frequency"]})

    for target_hz in target_freqs:
        stim_ids = [bid for bid, info in ssvep_info.items() if info["target_frequency"] == target_hz and bid in results]

        lines.append(f"## SSVEP at {target_hz} Hz")
        lines.append("")
        lines.append("SNR = power at target frequency / mean power in surrounding ±2 Hz noise band.")
        lines.append("")
        lines.append("| Channel | Stim SNR (dB) | Baseline SNR (dB) | Detection |")
        lines.append("|---------|---------------|-------------------|-----------|")

        for ch_idx, ch_name in enumerate(ch_names):
            snrs_stim = []
            for bid in stim_ids:
                r = results[bid]
                _, snr_db, _, _ = compute_ssvep_snr(r["freqs"], r["psd"][ch_idx], target_hz)
                snrs_stim.append(snr_db)

            snrs_base = []
            for bid in baseline_ids:
                if bid in results:
                    r = results[bid]
                    _, snr_db, _, _ = compute_ssvep_snr(r["freqs"], r["psd"][ch_idx], target_hz)
                    snrs_base.append(snr_db)

            stim = np.mean(snrs_stim) if snrs_stim else 0
            base = np.mean(snrs_base) if snrs_base else 0
            det = "Strong" if stim > 6 else "Weak" if stim > 3 else "None"
            lines.append(f"| {ch_name} | {stim:+.1f} | {base:+.1f} | {det} |")

        lines.append("")

    # Harmonic analysis (occipital focus)
    if "O1" in ch_names and "O2" in ch_names:
        o1, o2 = ch_names.index("O1"), ch_names.index("O2")

        lines.append("## Harmonic Analysis (O1/O2)")
        lines.append("")
        lines.append("SSVEP typically produces peaks at the fundamental frequency and its harmonics (2f, 3f).")
        lines.append("")
        lines.append("| Frequency | Harmonic | SNR (dB) |")
        lines.append("|-----------|----------|----------|")

        for target_hz in target_freqs:
            stim_ids = [bid for bid, info in ssvep_info.items() if info["target_frequency"] == target_hz and bid in results]
            for harmonic in [1, 2, 3]:
                freq_check = target_hz * harmonic
                if freq_check > 50:
                    break
                snrs = []
                for bid in stim_ids:
                    r = results[bid]
                    for ci in [o1, o2]:
                        _, snr_db, _, _ = compute_ssvep_snr(r["freqs"], r["psd"][ci], freq_check)
                        snrs.append(snr_db)
                avg_snr = np.mean(snrs) if snrs else 0
                label = f"{harmonic}f" if harmonic > 1 else "f"
                lines.append(f"| {target_hz} Hz | {label} = {freq_check} Hz | {avg_snr:+.1f} |")

        lines.append("")

    return "\n".join(lines)


## ── Alpha-specific report ───────────────────────────────────


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

    protocol = meta["protocol_id"]

    if protocol.startswith("p300"):
        print("P300 sessions use a separate analysis script:")
        print("  uv run python scripts/analyze_p300.py")
        print(f"  uv run python scripts/analyze_p300.py {meta['session_id']}")
        return

    if protocol.startswith("auditory-oddball"):
        print("Auditory oddball sessions use a separate analysis script:")
        print("  uv run python scripts/analyze_oddball.py")
        print(f"  uv run python scripts/analyze_oddball.py {meta['session_id']}")
        return

    is_ssvep = protocol.startswith("ssvep")
    ssvep_info = extract_ssvep_info(meta) if is_ssvep else {}

    if args.report:
        if is_ssvep:
            report = generate_ssvep_report(meta, eeg, blocks, results, ssvep_info, sr, ch_names)
        else:
            report = generate_report(meta, eeg, blocks, results, sr, ch_names)
        session_dir = SESSIONS_DIR / meta["session_id"]
        report_path = session_dir / "report.md"
        report_path.write_text(report)
        print(f"Report written to {report_path}")
    else:
        if is_ssvep:
            print_ssvep_analysis(results, ssvep_info, ch_names)
        else:
            print_analysis(results, ch_names)


if __name__ == "__main__":
    main()
