"""
Spatial mapping from 30-channel EAV (10-20) to 32-channel target layout.

EAV has excellent coverage: 27 of 30 channels are direct matches to the 32ch
target. Only 5 target channels need interpolation (FCz, TP9, TP10, CPz, AFz),
and 3 EAV channels (Fz, PO9, PO10) don't have direct target positions but
contribute to nearby interpolated targets.

Uses MNE montages to get 3D electrode positions and computes inverse-distance-
squared weighted (IDW) interpolation.

Run directly to recompute and print the mapping:
    uv run python -m data.eav.channel_mapping
"""

import numpy as np

from data.musicemo.songs import TARGET_CHANNELS

# The 30 EAV channels in recording order (already modern 10-10 names).
SOURCE_CHANNELS: list[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
]

RADIUS_MM = 30.0

# Precomputed mapping: target channel -> list of (source_index, normalized_weight).
# Generated from standard_1005 montage at 30mm radius with IDW(d^2).
# source_index refers to position in SOURCE_CHANNELS (0-29).
# 27/32 targets are direct 1:1 matches; 5 (FCz, TP9, TP10, CPz, AFz) interpolated.
EAV_TO_32CH_MAP: dict[str, list[tuple[int, float]]] = {
    "Fp1": [(0, 1.0000)],
    "F3": [(3, 1.0000)],
    "F7": [(2, 1.0000)],
    "FC5": [(7, 1.0000)],
    "FC1": [(8, 1.0000)],
    "FCz": [(8, 0.5037), (9, 0.4963)],
    "C3": [(12, 1.0000)],
    "T7": [(11, 1.0000)],
    "TP9": [(11, 0.5511), (20, 0.4489)],
    "CP5": [(16, 1.0000)],
    "CP1": [(17, 1.0000)],
    "Pz": [(22, 1.0000)],
    "P3": [(21, 1.0000)],
    "P7": [(20, 1.0000)],
    "O1": [(26, 1.0000)],
    "Oz": [(27, 1.0000)],
    "O2": [(28, 1.0000)],
    "P4": [(23, 1.0000)],
    "P8": [(24, 1.0000)],
    "TP10": [(15, 0.5372), (24, 0.4628)],
    "CP6": [(19, 1.0000)],
    "CP2": [(18, 1.0000)],
    "CPz": [(17, 0.5127), (22, 0.4873)],
    "Cz": [(13, 1.0000)],
    "C4": [(14, 1.0000)],
    "T8": [(15, 1.0000)],
    "FC6": [(10, 1.0000)],
    "FC2": [(9, 1.0000)],
    "F4": [(5, 1.0000)],
    "F8": [(6, 1.0000)],
    "AFz": [(4, 0.6482), (0, 0.3518)],
    "Fp2": [(1, 1.0000)],
}


def compute_mapping(
    radius_mm: float = RADIUS_MM,
) -> dict[str, list[tuple[int, float]]]:
    """Compute the 30ch→32ch spatial mapping using MNE montages."""
    import mne

    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos = montage.get_positions()["ch_pos"]

    source_positions = []
    for name in SOURCE_CHANNELS:
        if name not in ch_pos:
            raise ValueError(f"Source channel {name} not found in standard_1005 montage")
        source_positions.append(ch_pos[name])
    source_positions = np.array(source_positions)  # (30, 3)

    target_positions = []
    for name in TARGET_CHANNELS:
        if name not in ch_pos:
            raise ValueError(f"Target channel {name} not found in standard_1005 montage")
        target_positions.append(ch_pos[name])
    target_positions = np.array(target_positions)  # (32, 3)

    radius_m = radius_mm / 1000.0

    mapping: dict[str, list[tuple[int, float]]] = {}

    for t_idx, t_name in enumerate(TARGET_CHANNELS):
        t_pos = target_positions[t_idx]
        distances = np.linalg.norm(source_positions - t_pos, axis=1)  # (30,)

        within = np.where(distances <= radius_m)[0]

        if len(within) < 2:
            sorted_indices = np.argsort(distances)
            within = sorted_indices[:2]

        d = distances[within]
        if np.any(d == 0):
            weights = np.zeros_like(d)
            weights[d == 0] = 1.0
        else:
            weights = 1.0 / (d ** 2)

        weights = weights / weights.sum()

        mapping[t_name] = [
            (int(src_idx), float(w))
            for src_idx, w in zip(within, weights)
            if w > 0.001
        ]

        total = sum(w for _, w in mapping[t_name])
        mapping[t_name] = [(idx, w / total) for idx, w in mapping[t_name]]

    return mapping


def print_mapping(mapping: dict[str, list[tuple[int, float]]]):
    """Pretty-print the mapping with distances."""
    import mne

    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos = montage.get_positions()["ch_pos"]

    source_positions = np.array([ch_pos[n] for n in SOURCE_CHANNELS])

    print(f"{'Target':<8} {'#Src':>4}  Contributors (src_idx[name]: weight, dist_mm)")
    print("-" * 80)
    for t_name in TARGET_CHANNELS:
        contributors = mapping[t_name]
        t_pos = ch_pos[t_name]
        parts = []
        for src_idx, w in contributors:
            d = np.linalg.norm(source_positions[src_idx] - t_pos) * 1000
            parts.append(
                f"{src_idx}({SOURCE_CHANNELS[src_idx]}): {w:.3f} @ {d:.1f}mm"
            )
        print(f"{t_name:<8} {len(contributors):>4}  {', '.join(parts)}")


def format_as_constant(mapping: dict[str, list[tuple[int, float]]]) -> str:
    """Format mapping as a Python constant for hardcoding."""
    lines = ["EAV_TO_32CH_MAP: dict[str, list[tuple[int, float]]] = {"]
    for t_name in TARGET_CHANNELS:
        contributors = mapping[t_name]
        entries = ", ".join(f"({idx}, {w:.4f})" for idx, w in contributors)
        lines.append(f'    "{t_name}": [{entries}],')
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Computing 30-channel EAV → 32ch spatial mapping...\n")
    mapping = compute_mapping()
    print_mapping(mapping)
    print()
    print(format_as_constant(mapping))
