"""
Metadata for the HBN (Healthy Brain Network) movie-watching EEG dataset.

System: EGI HydroCel 129-channel (E1-E128 + Cz), 500 Hz
Task: Passive movie watching (4 video clips)
Subjects: ~2,639 (ages 5-21)
"""

from dataclasses import dataclass
from pathlib import Path

from data.songfam.channel_mapping import EGI_TO_32CH_MAP

# --- Paths ---

HBN_BIDS_DIR = Path("/kreka/research/willy/side/brain_datasets/HBN/BIDS_EEG")
HBN_VIDEO_DIR = Path("/kreka/research/willy/side/brain_datasets/hbn-video")
HBN_OUTPUT_DIR = Path("/kreka/research/willy/side/brain_datasets/nmed-processed")
HBN_AUDIO_DIR = HBN_OUTPUT_DIR / "hbn-audio"

# --- Recording parameters ---

HBN_SFREQ = 500  # Hz, native EEG sampling rate
TARGET_SFREQ = 125  # Hz, downsample target (matches NMED/SongFam pipeline)

# --- Releases ---

RELEASES = [
    "cmi_bids_R1",
    "cmi_bids_R2",
    "cmi_bids_R3",
    "cmi_bids_R4",
    "cmi_bids_R5",
    "cmi_bids_R6",
    "cmi_bids_R7",
    "cmi_bids_R8",
    "cmi_bids_R9",
    "cmi_bids_NC",
]


# --- Movie clip metadata ---


@dataclass
class Movie:
    task_name: str  # BIDS task name (e.g. "DespicableMe")
    title: str  # Human-readable title
    video_filename: str  # Filename in HBN_VIDEO_DIR
    trim_start: float  # Seconds to skip from start of video file
    trim_end: float | None  # Seconds to cut at (None = use full file)
    stimulus_duration: float  # Expected EEG stimulus duration (video_stop - video_start)


MOVIES = [
    Movie(
        task_name="DespicableMe",
        title="Despicable Me (Three Little Kittens)",
        video_filename="Three Little Kittens- Despicable Me [HNXxJIhVALI].mp4",
        trim_start=0.0,
        trim_end=None,
        stimulus_duration=170.55,
    ),
    Movie(
        task_name="FunwithFractals",
        title="Fun with Fractals",
        video_filename="Fun with Fractals [XwWyTts06tU].webm",
        trim_start=8.0,
        trim_end=171.0,  # 0:08 to 2:51
        stimulus_duration=163.00,
    ),
    Movie(
        task_name="DiaryOfAWimpyKid",
        title="Diary of a Wimpy Kid (Trailer)",
        video_filename="Diary of a Wimpy Kid Trailer [7ZVEIgPeDCE].webm",
        trim_start=0.0,
        trim_end=None,
        stimulus_duration=117.40,
    ),
    Movie(
        task_name="ThePresent",
        title="The Present",
        video_filename="The Present [152985022].mp4",
        trim_start=0.0,
        trim_end=203.07,  # Play up to credits
        stimulus_duration=203.07,
    ),
]

MOVIE_TASK_NAMES = [m.task_name for m in MOVIES]
MOVIE_BY_TASK = {m.task_name: m for m in MOVIES}

# --- Channel mapping ---

# HBN uses EGI HydroCel with 129 channels: E1-E128 + Cz.
# NMED uses the same system but with 125 channels: E1-E124 + Cz.
# The existing EGI_TO_32CH_MAP uses row indices into NMED's 125-channel layout,
# where Cz is at index 124. In HBN's 129-channel layout, Cz is at index 128.
# All other indices (0-123, mapping to E1-E124) are identical in both systems.
#
# We remap only the Cz index: 124 → 128.

_NMED_CZ_IDX = 124
_HBN_CZ_IDX = 128


def _remap_for_129ch() -> dict[str, list[tuple[int, float]]]:
    """Adapt EGI_TO_32CH_MAP from 125ch (NMED) to 129ch (HBN) layout."""
    remapped = {}
    for ch, contributors in EGI_TO_32CH_MAP.items():
        remapped[ch] = [
            (_HBN_CZ_IDX if idx == _NMED_CZ_IDX else idx, w)
            for idx, w in contributors
        ]
    return remapped


EGI129_TO_32CH_MAP = _remap_for_129ch()
