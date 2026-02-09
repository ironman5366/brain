"""
Spatial mapping from EGI HydroCel 128 (NMED) to 32-channel 10-20 (DS005876).

Uses MNE montages to get 3D electrode positions and computes inverse-distance-
squared weighted (IDW) interpolation from nearby EGI electrodes to each of the
32 target channels.

Run directly to recompute and print the mapping:
    uv run python -m data.songfam.channel_mapping
"""

import numpy as np

# The 32 DS005876 channels in recording order
TARGET_CHANNELS: list[str] = [
    "Fp1", "F3", "F7", "FC5", "FC1", "FCz", "C3", "T7",
    "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz",
    "O2", "P4", "P8", "TP10", "CP6", "CP2", "CPz", "Cz",
    "C4", "T8", "FC6", "FC2", "F4", "F8", "AFz", "Fp2",
]

# NMED EGI electrode names: E1-E124 + Cz (125 rows in the imputed .mat files).
# These match MNE's GSN-HydroCel-129 montage naming.
NMED_EGI_NAMES: list[str] = [
    f"E{i}" for i in range(1, 125)
] + ["Cz"]  # 125 total

RADIUS_MM = 25.0

# Precomputed mapping: target channel -> list of (nmed_row_index, normalized_weight).
# Generated from MNE GSN-HydroCel-129 and standard_1005 montages at 25mm radius with IDW(d^2).
EGI_TO_32CH_MAP: dict[str, list[tuple[int, float]]] = {
    "Fp1": [(20, 0.2476), (21, 0.2985), (24, 0.4539)],
    "F3": [(19, 0.1139), (23, 0.3725), (26, 0.2222), (27, 0.2914)],
    "F7": [(32, 1.0000)],
    "FC5": [(33, 0.6364), (39, 0.3636)],
    "FC1": [(12, 0.5396), (28, 0.1775), (29, 0.2829)],
    "FCz": [(5, 0.4209), (6, 0.2797), (105, 0.2994)],
    "C3": [(35, 0.2982), (40, 0.2660), (41, 0.4357)],
    "T7": [(44, 1.0000)],
    "TP9": [(56, 1.0000)],
    "CP5": [(45, 0.1321), (46, 0.1479), (50, 0.7200)],
    "CP1": [(52, 0.3029), (53, 0.4666), (60, 0.2305)],
    "Pz": [(61, 1.0000)],
    "P3": [(59, 1.0000)],
    "P7": [(57, 0.7442), (64, 0.2558)],
    "O1": [(69, 1.0000)],
    "Oz": [(74, 1.0000)],
    "O2": [(82, 1.0000)],
    "P4": [(84, 1.0000)],
    "P8": [(89, 0.2458), (95, 0.7542)],
    "TP10": [(99, 1.0000)],
    "CP6": [(96, 0.5863), (97, 0.2089), (101, 0.2049)],
    "CP2": [(77, 0.2181), (78, 0.3859), (85, 0.3960)],
    "CPz": [(54, 1.0000)],
    "Cz": [(30, 0.1028), (54, 0.2439), (79, 0.1100), (124, 0.5433)],
    "C4": [(92, 0.3826), (102, 0.3280), (103, 0.2893)],
    "T8": [(107, 1.0000)],
    "FC6": [(108, 0.3236), (115, 0.6764)],
    "FC2": [(104, 0.2777), (110, 0.2065), (111, 0.5158)],
    "F4": [(116, 0.2866), (122, 0.3151), (123, 0.3983)],
    "F8": [(0, 0.1067), (120, 0.1020), (121, 0.7913)],
    "AFz": [(9, 0.1913), (10, 0.2725), (15, 0.3508), (17, 0.1854)],
    "Fp2": [(7, 0.4595), (8, 0.2956), (13, 0.2449)],
}


def compute_mapping(
    radius_mm: float = RADIUS_MM,
) -> dict[str, list[tuple[int, float]]]:
    """Compute the EGI→32ch spatial mapping using MNE montages.

    Returns:
        Dict mapping each target channel name to a list of
        (nmed_row_index, normalized_weight) tuples.
    """
    import mne

    # Load EGI HydroCel montage (129-channel version includes Cz)
    egi_montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    egi_ch_pos = egi_montage.get_positions()["ch_pos"]

    # Load standard 10-05 montage for the 32 target channels
    std_montage = mne.channels.make_standard_montage("standard_1005")
    std_ch_pos = std_montage.get_positions()["ch_pos"]

    # Get 3D positions for each NMED EGI electrode (125 rows)
    egi_positions = []
    for name in NMED_EGI_NAMES:
        if name in egi_ch_pos:
            egi_positions.append(egi_ch_pos[name])
        else:
            raise ValueError(f"EGI electrode {name} not found in montage")
    egi_positions = np.array(egi_positions)  # (125, 3)

    # Get 3D positions for each target channel
    target_positions = []
    for name in TARGET_CHANNELS:
        if name in std_ch_pos:
            target_positions.append(std_ch_pos[name])
        else:
            raise ValueError(f"Target channel {name} not found in standard_1005 montage")
    target_positions = np.array(target_positions)  # (32, 3)

    # Compute distances and IDW mapping
    radius_m = radius_mm / 1000.0  # MNE uses meters
    mapping: dict[str, list[tuple[int, float]]] = {}

    for t_idx, t_name in enumerate(TARGET_CHANNELS):
        t_pos = target_positions[t_idx]
        distances = np.linalg.norm(egi_positions - t_pos, axis=1)  # (125,)

        # Find EGI electrodes within radius
        within = np.where(distances <= radius_m)[0]

        if len(within) == 0:
            raise ValueError(
                f"No EGI electrodes within {radius_mm}mm of {t_name}. "
                f"Nearest: E{np.argmin(distances)+1} at {distances.min()*1000:.1f}mm"
            )

        # Compute IDW weights
        d = distances[within]
        # Handle zero distance (exact match, e.g. Cz)
        if np.any(d == 0):
            weights = np.zeros_like(d)
            weights[d == 0] = 1.0
        else:
            weights = 1.0 / (d ** 2)

        # Normalize
        weights = weights / weights.sum()

        mapping[t_name] = [
            (int(egi_idx), float(w))
            for egi_idx, w in zip(within, weights)
            if w > 0.001  # drop negligible weights
        ]

        # Re-normalize after dropping
        total = sum(w for _, w in mapping[t_name])
        mapping[t_name] = [(idx, w / total) for idx, w in mapping[t_name]]

    return mapping


def print_mapping(mapping: dict[str, list[tuple[int, float]]]):
    """Pretty-print the mapping with distances."""
    import mne

    egi_montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    egi_ch_pos = egi_montage.get_positions()["ch_pos"]
    std_montage = mne.channels.make_standard_montage("standard_1005")
    std_ch_pos = std_montage.get_positions()["ch_pos"]

    egi_positions = np.array([egi_ch_pos[n] for n in NMED_EGI_NAMES])

    print(f"{'Target':<8} {'#EGI':>4}  Contributors (EGI_row: weight, dist_mm)")
    print("-" * 80)
    for t_name in TARGET_CHANNELS:
        contributors = mapping[t_name]
        t_pos = std_ch_pos[t_name]
        parts = []
        for egi_idx, w in contributors:
            d = np.linalg.norm(egi_positions[egi_idx] - t_pos) * 1000
            parts.append(f"row{egi_idx}({NMED_EGI_NAMES[egi_idx]}): {w:.3f} @ {d:.1f}mm")
        print(f"{t_name:<8} {len(contributors):>4}  {', '.join(parts)}")


def format_as_constant(mapping: dict[str, list[tuple[int, float]]]) -> str:
    """Format mapping as a Python constant for hardcoding."""
    lines = ["EGI_TO_32CH_MAP: dict[str, list[tuple[int, float]]] = {"]
    for t_name in TARGET_CHANNELS:
        contributors = mapping[t_name]
        entries = ", ".join(f"({idx}, {w:.4f})" for idx, w in contributors)
        lines.append(f'    "{t_name}": [{entries}],')
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Computing EGI HydroCel 128 → 32ch spatial mapping...\n")
    mapping = compute_mapping()
    print_mapping(mapping)
    print()
    print(format_as_constant(mapping))
