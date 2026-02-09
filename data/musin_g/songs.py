"""
Metadata for the OpenNeuro ds003774 MUSIN-G dataset.

Dataset: https://openneuro.org/datasets/ds003774
System: EGI HydroCel 128-channel (GSN-HydroCel-129), 250 Hz
Task: Passive music listening (12 songs of varied genres, eyes closed)
Subjects: 20 (Indian participants)
"""

from dataclasses import dataclass
from pathlib import Path


MUSIN_G_DATA_DIR = Path(
    "/kreka/research/willy/side/brain_datasets/openneuro_ds003774_musin_g"
)
MUSIN_G_OUTPUT_DIR = Path("/kreka/research/willy/side/brain_datasets/nmed-processed")
MUSIN_G_AUDIO_DIR = MUSIN_G_DATA_DIR / "Code" / "ESongs"
MUSIN_G_BEH_PATH = MUSIN_G_DATA_DIR / "stimuli" / "Behavioural_data"

MUSIN_G_SFREQ = 250  # Hz, native sampling rate
TARGET_SFREQ = 125  # Hz, downsample target (matches NMED/songfam)

SUBJECT_IDS = [f"sub-{i:03d}" for i in range(1, 21)]

# EGI channel names matching NMED convention: E1-E124 + Cz (125 total).
# The raw .set files have E1-E129; E125-E128 are face/neck electrodes
# and E129 = Cz reference. We drop E125-E128 and rename E129 -> Cz.
EGI_CHANNEL_NAMES = [f"E{i}" for i in range(1, 125)] + ["Cz"]


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    duration_sec: int
    tempo_bpm: float
    audio_filename: str


SONGS = [
    Song(1, "Trip to the Lonely Planet", "Mark Alow", "Deep House", 125, 121.95, "1.esh.wav"),
    Song(2, "Sail", "Awolnation", "Indie", 114, 119.0, "2.esh.wav"),
    Song(3, "Concept 15", "Kodomo", "Electronics", 132, 161.0, "3.esh.wav"),
    Song(4, "Aurore", "Claire David", "New Age", 111, 0.0, "4.esh.wav"),
    Song(5, "Proof", "Idiotape", "Electronic Dance", 124, 123.0, "5.esh.wav"),
    Song(6, "Glider", "Tycho", "Ambient", 100, 126.0, "6.esh.wav"),
    Song(7, "Raag Bihag", "B. Sivaramakrishna Rao", "Hindustani Classical", 116, 70.0, "7.esh.wav"),
    Song(8, "Albela Sajan", "Ismail Darbar", "Indian Semi-Classical", 121, 194.0, "8.esh.wav"),
    Song(9, "Mor Bani Thanghat Kare", "Sanjay Leela Bhansali", "Indian Folk", 126, 117.0, "9.esh.wav"),
    Song(10, "Fallin", "Dr. SaxLove", "Soft Jazz", 129, 197.0, "10.esh.wav"),
    Song(11, "Master of Running", "Rickeyabo", "Goth Rock", 113, 120.0, "11.esh.wav"),
    Song(12, "JB", "Nobody.one", "Progressive Instrumental Rock", 117, 146.0, "12.esh.wav"),
]

# Song lookup by ID
SONGS_BY_ID = {s.id: s for s in SONGS}
