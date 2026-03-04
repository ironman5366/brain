import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import soundfile as sf
import streamlit as st
import torch
from safetensors.torch import load_file

sys.path.append(str(Path(__file__).parent.parent))

from constants import STANDARD_CHANNELS

st.set_page_config(page_title="NMED-T Inspector", layout="wide")
st.title("NMED-T Inspector")
st.caption("Browse EEG windows from the NMED-T music listening dataset")

DATASET_DIR = Path("/kreka/research/willy/side/brain_datasets/nmed-processed")
AUDIO_DIR = DATASET_DIR / "audio"
COLOR_SIGNAL = "#1f77b4"
SFREQ = 125  # Hz


@st.cache_resource
def load_split(name: str):
    tensors_path = DATASET_DIR / f"nmed-{name}.safetensors"
    meta_path = DATASET_DIR / f"nmed-{name}-metadata.parquet"
    data = load_file(str(tensors_path))
    tensors = data["sparse_samples"]
    mask_indices = data["mask_indices"]
    metadata = pl.read_parquet(meta_path)
    return tensors, mask_indices, metadata


@st.cache_data
def load_audio(song_id: int) -> tuple[np.ndarray, int] | None:
    path = AUDIO_DIR / f"song_{song_id:02d}.flac"
    if not path.exists():
        return None
    audio, sr = sf.read(path)
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def get_audio_segment(song_id: int, start_sec: float, duration_sec: float = 1.0) -> bytes | None:
    result = load_audio(song_id)
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


# Partition selection
split = st.sidebar.radio("Split", ["train", "val"], index=0)
tensors, mask_indices, metadata = load_split(split)

channel_names = [STANDARD_CHANNELS[i] for i in mask_indices.tolist()]
n_samples = len(tensors)
n_channels = len(channel_names)
seq_len = tensors.shape[2]

st.sidebar.markdown(f"**{n_samples:,}** samples | **{n_channels}** ch | **{seq_len}** pts")

# Filters
st.sidebar.subheader("Filters")

song_names = sorted(metadata["song_name"].unique().to_list())
selected_songs = st.sidebar.multiselect("Song", song_names, default=[])

subject_ids = sorted(metadata["subject_id"].unique().to_list())
selected_subjects = st.sidebar.multiselect("Subject", subject_ids, default=[])

# Apply filters
filtered = metadata.with_row_index("_idx")
if selected_songs:
    filtered = filtered.filter(pl.col("song_name").is_in(selected_songs))
if selected_subjects:
    filtered = filtered.filter(pl.col("subject_id").is_in(selected_subjects))

total_filtered = len(filtered)
st.sidebar.caption(f"{total_filtered:,} / {n_samples:,} samples")

if total_filtered == 0:
    st.warning("No samples match the current filters.")
    st.stop()

# Navigation
if "sample_pos" not in st.session_state:
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
        import random
        st.session_state.sample_pos = random.randint(0, total_filtered - 1)
        st.rerun()
with col_nav4:
    pos = st.number_input(
        "Index",
        min_value=0,
        max_value=total_filtered - 1,
        key="sample_pos",
    )

row = filtered.row(st.session_state.sample_pos, named=True)
tensor_idx = row["_idx"]
sample = tensors[tensor_idx].numpy()  # (C, T)

# Layout: metadata + audio | EEG
col_meta, col_eeg = st.columns([1, 3])

with col_meta:
    st.subheader(f"{row['song_name']}")
    st.caption(f"by {row['artist']}")

    mc1, mc2 = st.columns(2)
    mc1.metric("Tempo", f"{row['tempo_bpm']} BPM")
    mc2.metric("Song #", row["song_id"])

    mc3, mc4 = st.columns(2)
    mc3.metric("Familiarity", f"{row['familiarity']}/9")
    mc4.metric("Enjoyment", f"{row['enjoyment']}/9")

    st.divider()
    st.metric("Subject", row["subject_id"])
    st.metric("Window", f"{row['window_idx']} ({row['window_start_sec']:.0f}s)")

    # Audio playback
    st.divider()
    audio_bytes = get_audio_segment(row["song_id"], row["window_start_sec"])
    if audio_bytes:
        st.caption(f"Audio: {row['window_start_sec']:.0f}s - {row['window_start_sec'] + 1:.0f}s")
        st.audio(audio_bytes, format="audio/wav")

        # Also offer a longer context (10s centered on this window)
        ctx_start = max(0, row["window_start_sec"] - 4.5)
        ctx_bytes = get_audio_segment(row["song_id"], ctx_start, 10.0)
        if ctx_bytes:
            st.caption(f"Context: {ctx_start:.0f}s - {ctx_start + 10:.0f}s")
            st.audio(ctx_bytes, format="audio/wav")
    else:
        st.caption("No audio file available")

with col_eeg:
    st.subheader("EEG Signal")

    # Channel selection
    n_display = st.slider("Channels to display", 1, n_channels, min(20, n_channels))
    display_channels = list(range(n_display))

    time = np.arange(seq_len) / SFREQ  # in seconds

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
