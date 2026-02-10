"""
Audio EEG Data Source Visualizer.

Unified browser across all audio-listening EEG datasets. Shows a pie chart
breakdown of samples per source and lets you browse individual samples
(EEG signal + audio playback) from any source.

Usage:
    uv run streamlit run viz/data_sources.py
"""

import random
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import soundfile as sf
import streamlit as st
from safetensors.torch import load_file

sys.path.append(str(Path(__file__).parent.parent))

from constants import STANDARD_CHANNELS
from data.musin_g.songs import MUSIN_G_AUDIO_DIR, SONGS_BY_ID as MUSING_SONGS_BY_ID
from data.songfam.songs import SONGFAM_STIMULI_DIR
from data.musicemo.songs import MUSICEMO_STIMULI_DIR

st.set_page_config(page_title="Audio EEG Data Sources", layout="wide")
st.title("Audio EEG Data Sources")
st.caption("Browse and compare EEG windows across all audio-listening datasets")

DATASET_DIR = Path("/kreka/research/willy/side/brain_datasets/nmed-processed")
SFREQ = 125  # Hz (all datasets standardized)
COLOR_SIGNAL = "#1f77b4"

# 32-channel target layout shared by musin-g, songfam, musicemo
DENSE_32_CHANNELS = [
    "Fp1", "F3", "F7", "FC5", "FC1", "FCz", "C3", "T7",
    "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz",
    "O2", "P4", "P8", "TP10", "CP6", "CP2", "CPz", "Cz",
    "C4", "T8", "FC6", "FC2", "F4", "F8", "AFz", "Fp2",
]


# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    name: str
    prefix: str
    is_sparse: bool
    audio_resolver: Callable[[dict], Path | None]
    stimulus_column: str
    display_columns: list[str]


def _nmed_audio(row: dict) -> Path | None:
    return DATASET_DIR / "audio" / f"song_{row['song_id']:02d}.flac"


def _musing_audio(row: dict) -> Path | None:
    song = MUSING_SONGS_BY_ID.get(row["song_id"])
    if song is None:
        return None
    return MUSIN_G_AUDIO_DIR / song.audio_filename


def _songfam_audio(row: dict) -> Path | None:
    return SONGFAM_STIMULI_DIR / row["song_filename"]


def _musicemo_audio(row: dict) -> Path | None:
    return MUSICEMO_STIMULI_DIR / row["mp3_filename"]


HBN_AUDIO_DIR = DATASET_DIR / "hbn-audio"


def _hbn_audio(row: dict) -> Path | None:
    return HBN_AUDIO_DIR / f"{row['task_name']}.wav"


SOURCES = [
    SourceConfig(
        name="NMED-T",
        prefix="nmed",
        is_sparse=True,
        audio_resolver=_nmed_audio,
        stimulus_column="song_name",
        display_columns=[
            "song_name", "artist", "tempo_bpm", "song_id",
            "familiarity", "enjoyment", "subject_id",
            "window_idx", "window_start_sec",
        ],
    ),
    SourceConfig(
        name="MUSIN-G",
        prefix="musin-g",
        is_sparse=False,
        audio_resolver=_musing_audio,
        stimulus_column="song_name",
        display_columns=[
            "song_name", "artist", "genre", "tempo_bpm", "song_id",
            "familiarity", "enjoyment", "subject_id",
            "window_idx", "window_start_sec",
        ],
    ),
    SourceConfig(
        name="Songfam",
        prefix="songfam",
        is_sparse=False,
        audio_resolver=_songfam_audio,
        stimulus_column="song_filename",
        display_columns=[
            "song_filename", "song_dur", "responded", "rt",
            "subject_id", "window_idx", "window_start_sec",
        ],
    ),
    SourceConfig(
        name="Musicemo",
        prefix="musicemo",
        is_sparse=False,
        audio_resolver=_musicemo_audio,
        stimulus_column="mp3_filename",
        display_columns=[
            "mp3_filename", "stim_code", "run",
            "subject_id", "window_idx", "window_start_sec",
        ],
    ),
    SourceConfig(
        name="HBN",
        prefix="hbn",
        is_sparse=False,
        audio_resolver=_hbn_audio,
        stimulus_column="movie_title",
        display_columns=[
            "movie_title", "task_name",
            "subject_id", "window_idx", "window_start_sec",
        ],
    ),
]

SOURCE_BY_NAME = {s.name: s for s in SOURCES}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_metadata_counts() -> dict[str, int]:
    counts = {}
    for src in SOURCES:
        total = 0
        for split in ("train", "val"):
            path = DATASET_DIR / f"{src.prefix}-{split}-metadata.parquet"
            if path.exists():
                total += pl.scan_parquet(path).select(pl.len()).collect().item()
        counts[src.name] = total
    return counts


@st.cache_resource
def load_split(prefix: str, split: str, is_sparse: bool):
    tensors_path = DATASET_DIR / f"{prefix}-{split}.safetensors"
    meta_path = DATASET_DIR / f"{prefix}-{split}-metadata.parquet"

    data = load_file(str(tensors_path))

    if is_sparse:
        tensors = data["sparse_samples"]
        mask_indices = data["mask_indices"]
        channel_names = [STANDARD_CHANNELS[i] for i in mask_indices.tolist()]
    else:
        tensors = data["samples"]
        channel_names = list(DENSE_32_CHANNELS)

    metadata = pl.read_parquet(meta_path)
    return tensors, channel_names, metadata


@st.cache_data
def load_audio_file(path_str: str) -> tuple[np.ndarray, int] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def get_audio_segment(audio_path: Path, start_sec: float, duration_sec: float = 1.0) -> bytes | None:
    result = load_audio_file(str(audio_path))
    if result is None:
        return None
    audio, sr = result
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    segment = audio[start_sample:end_sample]
    if len(segment) == 0:
        return None
    buf = BytesIO()
    sf.write(buf, segment, sr, format="WAV")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Overview: pie chart + summary table
# ---------------------------------------------------------------------------

st.subheader("Dataset Overview")

counts = load_metadata_counts()
total_all = sum(counts.values())

col_pie, col_table = st.columns([1, 1])

with col_pie:
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title(f"Total: {total_all:,} samples")
    st.pyplot(fig)
    plt.close()

with col_table:
    summary_rows = []
    for src in SOURCES:
        count = counts[src.name]
        summary_rows.append({
            "Source": src.name,
            "Samples": f"{count:,}",
            "Share": f"{count / total_all * 100:.1f}%",
            "Prefix": src.prefix,
        })
    st.dataframe(pl.from_dicts(summary_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar: source + split selection
# ---------------------------------------------------------------------------

st.sidebar.header("Browse Samples")

source_name = st.sidebar.selectbox("Source", [s.name for s in SOURCES])
source = SOURCE_BY_NAME[source_name]

split = st.sidebar.radio("Split", ["train", "val"], index=0)


# ---------------------------------------------------------------------------
# Load data + filters
# ---------------------------------------------------------------------------

st.divider()
st.subheader(f"{source.name} \u2014 {split}")

tensors, channel_names, metadata = load_split(source.prefix, split, source.is_sparse)
n_samples = len(tensors)
n_channels = len(channel_names)
seq_len = tensors.shape[2]

st.sidebar.markdown(f"**{n_samples:,}** samples | **{n_channels}** ch | **{seq_len}** pts")

st.sidebar.subheader("Filters")

stim_col = source.stimulus_column
stim_values = sorted(metadata[stim_col].unique().to_list())
selected_stims = st.sidebar.multiselect(
    stim_col.replace("_", " ").title(), stim_values, default=[]
)

subject_ids = sorted(metadata["subject_id"].unique().to_list())
selected_subjects = st.sidebar.multiselect("Subject", subject_ids, default=[])

filtered = metadata.with_row_index("_idx")
if selected_stims:
    filtered = filtered.filter(pl.col(stim_col).is_in(selected_stims))
if selected_subjects:
    filtered = filtered.filter(pl.col("subject_id").is_in(selected_subjects))

total_filtered = len(filtered)
st.sidebar.caption(f"{total_filtered:,} / {n_samples:,} samples")

if total_filtered == 0:
    st.warning("No samples match the current filters.")
    st.stop()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

if "sample_pos" not in st.session_state:
    st.session_state.sample_pos = 0

if st.session_state.sample_pos >= total_filtered:
    st.session_state.sample_pos = 0

col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 1, 1])
with col_nav1:
    if st.button("Prev") and st.session_state.sample_pos > 0:
        st.session_state.sample_pos -= 1
        st.rerun()
with col_nav2:
    if st.button("Next") and st.session_state.sample_pos < total_filtered - 1:
        st.session_state.sample_pos += 1
        st.rerun()
with col_nav3:
    if st.button("Random"):
        st.session_state.sample_pos = random.randint(0, total_filtered - 1)
        st.rerun()
with col_nav4:
    st.number_input(
        "Index",
        min_value=0,
        max_value=total_filtered - 1,
        key="sample_pos",
    )


# ---------------------------------------------------------------------------
# Sample view: metadata + audio | EEG
# ---------------------------------------------------------------------------

row = filtered.row(st.session_state.sample_pos, named=True)
tensor_idx = row["_idx"]
sample = tensors[tensor_idx].numpy()  # (C, T)

col_meta, col_eeg = st.columns([1, 3])

with col_meta:
    # Adaptive header
    if "song_name" in row and row.get("song_name") is not None:
        st.subheader(row["song_name"])
        if "artist" in row and row.get("artist") is not None:
            st.caption(f"by {row['artist']}")
    elif "movie_title" in row:
        st.subheader(row["movie_title"])
    elif "song_filename" in row:
        st.subheader(row["song_filename"])
    elif "mp3_filename" in row:
        st.subheader(row["mp3_filename"])

    # Adaptive metrics from display_columns
    header_cols = {"song_name", "artist"}
    display_cols = [c for c in source.display_columns if c in row and c not in header_cols]

    for i in range(0, len(display_cols), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(display_cols):
                key = display_cols[idx]
                val = row[key]
                label = key.replace("_", " ").title()
                if key == "familiarity" and val is not None:
                    col.metric(label, f"{val}/9")
                elif key == "enjoyment" and val is not None:
                    col.metric(label, f"{val}/9")
                elif key == "tempo_bpm" and val is not None:
                    col.metric(label, f"{val} BPM")
                elif key == "window_start_sec" and val is not None:
                    col.metric("Window", f"{row.get('window_idx', '?')} ({val:.0f}s)")
                elif key != "window_idx":
                    col.metric(label, val)

    # Audio playback
    st.divider()
    audio_path = source.audio_resolver(row)
    if audio_path and audio_path.exists():
        window_start = row.get("window_start_sec", 0.0)

        audio_bytes = get_audio_segment(audio_path, window_start, 1.0)
        if audio_bytes:
            st.caption(f"Audio: {window_start:.0f}s \u2013 {window_start + 1:.0f}s")
            st.audio(audio_bytes, format="audio/wav")

        ctx_start = max(0, window_start - 4.5)
        ctx_bytes = get_audio_segment(audio_path, ctx_start, 10.0)
        if ctx_bytes:
            st.caption(f"Context: {ctx_start:.0f}s \u2013 {ctx_start + 10:.0f}s")
            st.audio(ctx_bytes, format="audio/wav")
    else:
        st.caption("No audio file available")

with col_eeg:
    st.subheader("EEG Signal")

    n_display = st.slider("Channels to display", 1, n_channels, min(20, n_channels))
    display_channels = list(range(n_display))

    time = np.arange(seq_len) / SFREQ

    fig, axes = plt.subplots(
        len(display_channels), 1,
        figsize=(14, len(display_channels) * 0.6),
        sharex=True,
    )
    if len(display_channels) == 1:
        axes = [axes]

    for i, ch_idx in enumerate(display_channels):
        ax = axes[i]
        ax.plot(time, sample[ch_idx], linewidth=0.5, color=COLOR_SIGNAL)
        ax.set_ylabel(channel_names[ch_idx], fontsize=7, rotation=0, ha="right")
        ax.tick_params(axis="y", labelsize=5)
        ax.set_xlim(0, seq_len / SFREQ)
        ax.grid(True, alpha=0.2)
        ax.set_yticks([])

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
