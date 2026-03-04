from pathlib import Path
from dataclasses import dataclass

NMED_DATA_DIR = Path("/kreka/research/willy/side/brain_datasets/nmed")
NMED_MUSIC_DIR = Path("/kreka/research/willy/side/brain_datasets/nmed_music")
NMED_OUTPUT_DIR = Path("/kreka/research/willy/side/brain_datasets/nmed-processed")
NMED_AUDIO_DIR = NMED_OUTPUT_DIR / "audio"

NMED_SFREQ = 125  # Hz, native sample rate of imputed data


@dataclass
class Song:
    id: int  # 1-10
    file_number: int  # 21-30 (as in song21_Imputed.mat)
    title: str
    artist: str
    tempo_bpm: float
    asin: str
    audio_source: str  # filename or path within NMED_MUSIC_DIR


SONGS = [
    Song(1, 21, "First Fires", "Bonobo", 55.97, "B00CJE73J6",
         "Bonobo - First Fires/01 Bonobo - First Fires.flac"),
    Song(2, 22, "Oino", "LA Priest", 69.44, "B00T4NHS2W",
         "LA Priest - Oino.zip:01 LA Priest - Oino.flac"),
    Song(3, 23, "Tiptoes", "Daedelus", 74.26, "B011SAZRLC",
         "Daedelus - Tiptoes.flac"),
    Song(4, 24, "Careless Love", "Croquet Club", 82.42, "B06X9736NJ",
         "Croquet Club - Careless Love.flac"),
    Song(5, 25, "Lebanese Blonde", "Thievery Corporation", 91.46, "B000SF16MI",
         "Thievery Corporation - Lebanese Blonde.flac"),
    Song(6, 26, "Canopee", "Polo & Pan", 96.15, "B01GOL4IB0",
         "Polo & Pan - Canopée.zip:01 Polo & Pan - Canopée.flac"),
    Song(7, 27, "Doing Yoga", "Kazy Lambist", 108.70, "B01JDDVIQ4",
         "Kazy Lambist - Doing Yoga.flac"),
    Song(8, 28, "Until the Sun Needs to Rise", "Rufus du Sol", 120.00, "B01APT6JKA",
         "RÜFÜS DU SOL - Until the Sun Needs to Rise.flac"),
    Song(9, 29, "Silent Shout", "The Knife", 128.21, "B00IMN40O4",
         "The Knife - Silent Shout.flac"),
    Song(10, 30, "The Last Thing You Should Do", "David Bowie", 150.00, "B018GS2A46",
         "David Bowie - The Last Thing You Should Do.flac"),
]

SONG_BY_FILE_NUMBER = {s.file_number: s for s in SONGS}
SONG_BY_ID = {s.id: s for s in SONGS}

# Subject IDs in the dataset (20 participants, no S01 or S18, S22)
SUBJECT_IDS = [
    "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10",
    "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S19", "S20",
    "S21", "S23",
]

# EGI HydroCel 128 electrode → nearest standard 10-5 channel name.
# One entry per row of the imputed data (125 rows: E1-E124 + Cz).
# Computed from MNE montage nearest-neighbor matching.
# Some nearby electrodes map to the same standard position (5 duplicates) —
# standardize_epochs() handles this by overwriting, which is fine.
EGI_CHANNEL_NAMES: list[str] = [
    "AFF10H",  # row 0  = E1
    "AF8H",    # row 1  = E2
    "AF4",     # row 2  = E3
    "AFF2",    # row 3  = E4
    "F2H",     # row 4  = E5
    "FFCZ",    # row 5  = E6
    "FCC1H",   # row 6  = E7
    "FP2",     # row 7  = E8
    "AFP4H",   # row 8  = E9
    "AF2",     # row 9  = E10
    "AFFZ",    # row 10 = E11
    "F1H",     # row 11 = E12
    "FC1",     # row 12 = E13
    "FPZ",     # row 13 = E14
    "AFP1",    # row 14 = E15
    "AFZ",     # row 15 = E16
    "FPZ",     # row 16 = E17 (dup)
    "AF1",     # row 17 = E18
    "AFF1",    # row 18 = E19
    "F3H",     # row 19 = E20
    "FPZ",     # row 20 = E21 (dup)
    "AFP3H",   # row 21 = E22
    "AF3",     # row 22 = E23
    "F3",      # row 23 = E24
    "FP1",     # row 24 = E25
    "AF7H",    # row 25 = E26
    "F5H",     # row 26 = E27
    "FFC3",    # row 27 = E28
    "FC3H",    # row 28 = E29
    "FCC1",    # row 29 = E30
    "C1H",     # row 30 = E31
    "AFF9H",   # row 31 = E32
    "F7H",     # row 32 = E33
    "FFC5",    # row 33 = E34
    "FC5H",    # row 34 = E35
    "FCC3",    # row 35 = E36
    "C1",      # row 36 = E37
    "F9H",     # row 37 = E38
    "FT7",     # row 38 = E39
    "FCC5",    # row 39 = E40
    "C5H",     # row 40 = E41
    "CCP3",    # row 41 = E42
    "F9",      # row 42 = E43
    "FT9H",    # row 43 = E44
    "T7",      # row 44 = E45
    "CCP5",    # row 45 = E46
    "CCP5H",   # row 46 = E47
    "F9",      # row 47 = E48 (dup)
    "FT9",     # row 48 = E49
    "TP7",     # row 49 = E50
    "CP5",     # row 50 = E51
    "CP5H",    # row 51 = E52
    "CP3H",    # row 52 = E53
    "CCP1H",   # row 53 = E54
    "CCPZ",    # row 54 = E55
    "A1",      # row 55 = E56
    "TP9H",    # row 56 = E57
    "P7",      # row 57 = E58
    "P5",      # row 58 = E59
    "CPP3",    # row 59 = E60
    "CPP1",    # row 60 = E61
    "PZ",      # row 61 = E62
    "M1",      # row 62 = E63
    "P9H",     # row 63 = E64
    "PPO7",    # row 64 = E65
    "PPO5H",   # row 65 = E66
    "PPO1",    # row 66 = E67
    "PPO9",    # row 67 = E68
    "PO9",     # row 68 = E69
    "POO7",    # row 69 = E70
    "PO1",     # row 70 = E71
    "POZ",     # row 71 = E72
    "POO9",    # row 72 = E73
    "I1",      # row 73 = E74
    "OZ",      # row 74 = E75
    "PO2",     # row 75 = E76
    "PPO2",    # row 76 = E77
    "CPP2",    # row 77 = E78
    "CCP2H",   # row 78 = E79
    "C2H",     # row 79 = E80
    "IZ",      # row 80 = E81
    "I2",      # row 81 = E82
    "POO8",    # row 82 = E83
    "PPO6H",   # row 83 = E84
    "P4H",     # row 84 = E85
    "CP4H",    # row 85 = E86
    "C2",      # row 86 = E87
    "POO10",   # row 87 = E88
    "PO10",    # row 88 = E89
    "PPO8",    # row 89 = E90
    "P6",      # row 90 = E91
    "CP4",     # row 91 = E92
    "CCP4",    # row 92 = E93
    "PPO10",   # row 93 = E94
    "P10H",    # row 94 = E95
    "P8",      # row 95 = E96
    "CPP6",    # row 96 = E97
    "CCP6H",   # row 97 = E98
    "M2",      # row 98 = E99
    "TP10H",   # row 99 = E100
    "TP8",     # row 100 = E101
    "CCP6",    # row 101 = E102
    "C6H",     # row 102 = E103
    "FCC4H",   # row 103 = E104
    "FCC2",    # row 104 = E105
    "FCC2H",   # row 105 = E106
    "A2",      # row 106 = E107
    "T8",      # row 107 = E108
    "FCC6",    # row 108 = E109
    "FC4",     # row 109 = E110
    "FC4H",    # row 110 = E111
    "FC2H",    # row 111 = E112
    "FT10",    # row 112 = E113
    "FT10H",   # row 113 = E114
    "FT8",     # row 114 = E115
    "FFC6",    # row 115 = E116
    "FFC4",    # row 116 = E117
    "F4H",     # row 117 = E118
    "F10",     # row 118 = E119
    "F10",     # row 119 = E120 (dup)
    "F10H",    # row 120 = E121
    "F8H",     # row 121 = E122
    "F6H",     # row 122 = E123
    "F4H",     # row 123 = E124 (dup)
    "CZ",      # row 124 = Cz (vertex reference)
]
