"""
Metadata for the OpenNeuro DS002721 Music Emotion dataset.

Dataset: https://openneuro.org/datasets/ds002721
Paper: Daly et al. (2018) - "Neural and physiological data from participants
       listening to affective music", Scientific Data.
System: 19-channel EEG (standard 10-20), 1000 Hz
Task: Listening to 12s film score excerpts with emotion ratings (8 dimensions)
Stimuli: Eerola & Vuoskoski (2010) film score excerpts (91 unique clips used)
"""

from pathlib import Path

MUSICEMO_DATA_DIR = Path(
    "/kreka/research/willy/side/brain_datasets/openneuro_ds002721_music_emotion"
)
MUSICEMO_OUTPUT_DIR = Path("/kreka/research/willy/side/brain_datasets/nmed-processed")
MUSICEMO_STIMULI_DIR = MUSICEMO_DATA_DIR / "stimuli" / "Set1"

MUSICEMO_SFREQ = 1000  # Hz, native sampling rate
TARGET_SFREQ = 125  # Hz, downsample target (matches NMED and DS005876)

# All 31 subject IDs
SUBJECT_IDS = [f"sub-{i:02d}" for i in range(1, 32)]

# 19-channel names in recording order (standard 10-20 system)
CHANNEL_NAMES = [
    "FP1", "FP2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

# Mapping from old 10-20 names (used in this dataset) to modern 10-10 names
# (used in MNE standard montages). Needed for spatial coordinate lookups.
OLD_TO_MODERN_NAMES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "FP1": "Fp1",
    "FP2": "Fp2",
}

# Music listening runs (runs 1 and 6 are resting state)
MUSIC_RUNS = [2, 3, 4, 5]

# Event codes: stimulus ID codes span 302-657 (the hundreds digit varies by
# trial; the clip number is always code % 100). The documented range is 301-360
# but this only covers ~31% of actual codes.
# Code 788 = music playback start, 786 = fixation cross.
EVENT_MUSIC_START = 788

# 32-channel target layout (same as DS005876 / songfam)
TARGET_CHANNELS = [
    "Fp1", "F3", "F7", "FC5", "FC1", "FCz", "C3", "T7",
    "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz",
    "O2", "P4", "P8", "TP10", "CP6", "CP2", "CPz", "Cz",
    "C4", "T8", "FC6", "FC2", "F4", "F8", "AFz", "Fp2",
]
