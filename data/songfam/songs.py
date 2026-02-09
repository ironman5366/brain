"""
Metadata for the OpenNeuro DS005876 Song Familiarity dataset.

Dataset: https://openneuro.org/datasets/ds005876
System: Brain Products actiCHamp 32-channel, 1000 Hz
Task: Song familiarity detection (121 short melody snippets, 5-17s each)
"""

from pathlib import Path

SONGFAM_DATA_DIR = Path(
    "/kreka/research/willy/side/brain_datasets/openneuro_ds005876_song_familiarity"
)
SONGFAM_OUTPUT_DIR = Path("/kreka/research/willy/side/brain_datasets/nmed-processed")
SONGFAM_STIMULI_DIR = SONGFAM_DATA_DIR / "stimuli"

SONGFAM_SFREQ = 1000  # Hz, native sampling rate
TARGET_SFREQ = 125  # Hz, downsample target (matches NMED)

# All 29 subject IDs (sub-08 is missing from the dataset)
SUBJECT_IDS = [f"sub-{i:02d}" for i in range(1, 31) if i != 8]

# 32-channel names in recording order
CHANNEL_NAMES = [
    "Fp1", "F3", "F7", "FC5", "FC1", "FCz", "C3", "T7",
    "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz",
    "O2", "P4", "P8", "TP10", "CP6", "CP2", "CPz", "Cz",
    "C4", "T8", "FC6", "FC2", "F4", "F8", "AFz", "Fp2",
]
