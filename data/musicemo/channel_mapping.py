"""
Spatial mapping from 19-channel 10-20 (DS002721) to 32-channel 10-20 target.

NOTE: This is a TEMPORARY coarse mapping. With only 19 source electrodes,
the interpolation for the 14 channels without a direct match (FC5, FC1, FCz,
TP9, CP5, CP1, Oz, TP10, CP6, CP2, CPz, FC6, FC2, AFz) is much less precise
than the NMED 120→32 mapping. A more principled approach (e.g. spherical
splines, or reducing to a common channel subset) is planned for later.

Uses MNE montages to get 3D electrode positions and computes inverse-distance-
squared weighted (IDW) interpolation from nearby source electrodes to each of
the 32 target channels. Uses a wider 40mm radius than the songfam mapping
(25mm) to compensate for the sparser source set.

Run directly to recompute and print the mapping:
    uv run python -m data.musicemo.channel_mapping
"""

import numpy as np

from data.musicemo.songs import TARGET_CHANNELS

# The 19 DS002721 channels in recording order, using modern names for MNE lookup.
# Original names: FP1→Fp1, FP2→Fp2, T3→T7, T4→T8, T5→P7, T6→P8
SOURCE_CHANNELS_MODERN: list[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]

RADIUS_MM = 40.0

# Precomputed mapping: target channel -> list of (source_index, normalized_weight).
# Generated from standard_1005 montage at 40mm radius with IDW(d^2).
# source_index refers to position in SOURCE_CHANNELS_MODERN (0-18).
#
# TEMPORARY: Coarse mapping from 19 sources. See module docstring.
TENTWENTY_TO_32CH_MAP: dict[str, list[tuple[int, float]]] = {
    "Fp1": [(0, 1.0000)],
    "F3": [(3, 1.0000)],
    "F7": [(2, 1.0000)],
    "FC5": [(2, 0.5395), (3, 0.4605)],
    "FC1": [(4, 0.5004), (3, 0.4996)],
    "FCz": [(4, 0.5015), (9, 0.4985)],
    "C3": [(8, 1.0000)],
    "T7": [(7, 1.0000)],
    "TP9": [(7, 0.5511), (12, 0.4489)],
    "CP5": [(12, 0.5560), (13, 0.4440)],
    "CP1": [(14, 0.5045), (13, 0.4955)],
    "Pz": [(14, 1.0000)],
    "P3": [(13, 1.0000)],
    "P7": [(12, 1.0000)],
    "O1": [(17, 1.0000)],
    "Oz": [(17, 0.5040), (18, 0.4960)],
    "O2": [(18, 1.0000)],
    "P4": [(15, 1.0000)],
    "P8": [(16, 1.0000)],
    "TP10": [(11, 0.5372), (16, 0.4628)],
    "CP6": [(16, 0.5553), (15, 0.4447)],
    "CP2": [(15, 0.5212), (14, 0.4788)],
    "CPz": [(9, 0.4946), (14, 0.5054)],
    "Cz": [(9, 1.0000)],
    "C4": [(10, 1.0000)],
    "T8": [(11, 1.0000)],
    "FC6": [(6, 0.5295), (5, 0.4705)],
    "FC2": [(4, 0.5145), (5, 0.4855)],
    "F4": [(5, 1.0000)],
    "F8": [(6, 1.0000)],
    "AFz": [(4, 0.6482), (0, 0.3518)],
    "Fp2": [(1, 1.0000)],
}


def compute_mapping(
    radius_mm: float = RADIUS_MM,
) -> dict[str, list[tuple[int, float]]]:
    """Compute the 19ch→32ch spatial mapping using MNE montages.

    Returns:
        Dict mapping each target channel name to a list of
        (source_index, normalized_weight) tuples.
    """
    import mne

    # Load standard 10-05 montage (contains both source and target positions)
    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos = montage.get_positions()["ch_pos"]

    # Get 3D positions for each source channel (19 channels)
    source_positions = []
    for name in SOURCE_CHANNELS_MODERN:
        if name not in ch_pos:
            raise ValueError(f"Source channel {name} not found in standard_1005 montage")
        source_positions.append(ch_pos[name])
    source_positions = np.array(source_positions)  # (19, 3)

    # Get 3D positions for each target channel (32 channels)
    target_positions = []
    for name in TARGET_CHANNELS:
        if name not in ch_pos:
            raise ValueError(f"Target channel {name} not found in standard_1005 montage")
        target_positions.append(ch_pos[name])
    target_positions = np.array(target_positions)  # (32, 3)

    radius_m = radius_mm / 1000.0  # MNE uses meters

    mapping: dict[str, list[tuple[int, float]]] = {}

    for t_idx, t_name in enumerate(TARGET_CHANNELS):
        t_pos = target_positions[t_idx]
        distances = np.linalg.norm(source_positions - t_pos, axis=1)  # (19,)

        # Find source electrodes within radius
        within = np.where(distances <= radius_m)[0]

        # If no sources within radius, expand until we find at least 2
        if len(within) < 2:
            sorted_indices = np.argsort(distances)
            within = sorted_indices[:2]

        # Compute IDW weights
        d = distances[within]
        # Handle zero distance (exact match)
        if np.any(d == 0):
            weights = np.zeros_like(d)
            weights[d == 0] = 1.0
        else:
            weights = 1.0 / (d ** 2)

        # Normalize
        weights = weights / weights.sum()

        mapping[t_name] = [
            (int(src_idx), float(w))
            for src_idx, w in zip(within, weights)
            if w > 0.001  # drop negligible weights
        ]

        # Re-normalize after dropping
        total = sum(w for _, w in mapping[t_name])
        mapping[t_name] = [(idx, w / total) for idx, w in mapping[t_name]]

    return mapping


def print_mapping(mapping: dict[str, list[tuple[int, float]]]):
    """Pretty-print the mapping with distances."""
    import mne

    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos = montage.get_positions()["ch_pos"]

    source_positions = np.array([ch_pos[n] for n in SOURCE_CHANNELS_MODERN])

    print(f"{'Target':<8} {'#Src':>4}  Contributors (src_idx[name]: weight, dist_mm)")
    print("-" * 80)
    for t_name in TARGET_CHANNELS:
        contributors = mapping[t_name]
        t_pos = ch_pos[t_name]
        parts = []
        for src_idx, w in contributors:
            d = np.linalg.norm(source_positions[src_idx] - t_pos) * 1000
            parts.append(
                f"{src_idx}({SOURCE_CHANNELS_MODERN[src_idx]}): {w:.3f} @ {d:.1f}mm"
            )
        print(f"{t_name:<8} {len(contributors):>4}  {', '.join(parts)}")


def format_as_constant(mapping: dict[str, list[tuple[int, float]]]) -> str:
    """Format mapping as a Python constant for hardcoding."""
    lines = ["TENTWENTY_TO_32CH_MAP: dict[str, list[tuple[int, float]]] = {"]
    for t_name in TARGET_CHANNELS:
        contributors = mapping[t_name]
        entries = ", ".join(f"({idx}, {w:.4f})" for idx, w in contributors)
        lines.append(f'    "{t_name}": [{entries}],')
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Computing 19-channel 10-20 → 32ch spatial mapping...\n")
    mapping = compute_mapping()
    print_mapping(mapping)
    print()
    print(format_as_constant(mapping))
