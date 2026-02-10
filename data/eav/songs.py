"""
Metadata for the EAV (EEG-Audio-Video) dataset.

Dataset: https://zenodo.org/records/13799131
Paper: Lee et al. (2024) - "EAV: EEG-Audio-Video Dataset for Emotion
       Recognition in Conversational Contexts", Scientific Data.
System: 30-channel EEG (BrainAmp, Brain Products), 500 Hz
Task: Cue-based conversation with 5 emotions (neutral, anger, happiness,
      sadness, calmness) in listening and speaking conditions.
Subjects: 42 participants, 200 trials each (8,400 total interactions).
"""

from pathlib import Path

EAV_DATA_DIR = Path("/kreka/research/willy/side/brain_datasets/eav/EAV")
EAV_OUTPUT_DIR = Path("/kreka/research/willy/side/brain_datasets/nmed-processed")
EAV_AUDIO_DIR = EAV_OUTPUT_DIR / "eav-audio"

EAV_SFREQ = 500  # Hz, native sampling rate
TARGET_SFREQ = 125  # Hz, downsample target (matches NMED, MUSIN-G, etc.)

# 42 subjects
SUBJECT_IDS = [f"subject{i}" for i in range(1, 43)]

# 30 EEG channel names in recording order (from paper, standard 10-20/10-10).
# Reference: mastoids, Ground: AFz.
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
]

# Label layout: one-hot (10, N_TRIALS) where rows map to:
LABEL_NAMES = [
    "neutral_listening",   # 0
    "neutral_speaking",    # 1
    "sadness_listening",   # 2
    "sadness_speaking",    # 3
    "anger_listening",     # 4
    "anger_speaking",      # 5
    "happiness_listening", # 6
    "happiness_speaking",  # 7
    "calmness_listening",  # 8
    "calmness_speaking",   # 9
]

EMOTION_NAMES = {
    0: "neutral",
    1: "neutral",
    2: "sadness",
    3: "sadness",
    4: "anger",
    5: "anger",
    6: "happiness",
    7: "happiness",
    8: "calmness",
    9: "calmness",
}

CONDITION_NAMES = {
    0: "listening",
    1: "speaking",
    2: "listening",
    3: "speaking",
    4: "listening",
    5: "speaking",
    6: "listening",
    7: "speaking",
    8: "listening",
    9: "speaking",
}
