"""
Audio Embed Eval — decode EEG→Audio Embedding predictions and listen to them.

Usage:
    uv run streamlit run viz/audio_embed_eval.py
"""

import random
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import soundfile as sf
import streamlit as st
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import EncodecModel

sys.path.append(str(Path(__file__).parent.parent))

from constants import STANDARD_CHANNELS
from data.nmed.build_encodec import ENCODEC_SFREQ, FRAMES_PER_SECOND, load_audio_24khz
from data.nmed.songs import NMED_AUDIO_DIR, NMED_OUTPUT_DIR, SONG_BY_ID
from models.audio_embed import EEGAudioEmbed

st.set_page_config(page_title="Audio Embed Eval", layout="wide")
st.title("Audio Embed Eval")
st.caption("EEG → Audio Embedding: decode predictions and listen to reconstructed audio")

DATASET_DIR = NMED_OUTPUT_DIR
CHECKPOINTS_DIR = Path("/kreka/research/willy/side/brain-worktrees/nmed/checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EEG_SFREQ = 125  # Hz

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def load_encodec():
    """Load EnCodec model for decoding embeddings → audio."""
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    model.eval()
    model.to(DEVICE)
    return model


@st.cache_resource
def load_eeg_model(checkpoint_path: str):
    """Load an EEGAudioEmbed checkpoint."""
    model = EEGAudioEmbed.from_pretrained(checkpoint_path)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_split_data(split: str):
    """Load EEG tensors, audio embeddings, and metadata for a split."""
    eeg_data = load_file(str(DATASET_DIR / f"nmed-{split}.safetensors"))
    encodec_data = load_file(str(DATASET_DIR / f"nmed-{split}-encodec.safetensors"))
    metadata = pl.read_parquet(DATASET_DIR / f"nmed-{split}-metadata.parquet")
    return (
        eeg_data["sparse_samples"],  # (N, 120, 125)
        eeg_data["mask_indices"],  # channel mask indices
        encodec_data["audio_embeds"],  # (N, 128, 75)
        metadata,
    )


@st.cache_data
def load_song_audio(song_id: int) -> np.ndarray:
    """Load song audio resampled to 24 kHz, as a numpy array."""
    return load_audio_24khz(song_id).numpy()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def decode_embeddings(encodec_model, embeddings: torch.Tensor) -> np.ndarray:
    """Decode EnCodec embeddings to audio waveform.

    Args:
        encodec_model: HuggingFace EncodecModel
        embeddings: (128, 75) single-sample embeddings

    Returns:
        audio: (24000,) numpy array at 24 kHz
    """
    with torch.no_grad():
        emb = embeddings.unsqueeze(0).to(DEVICE)  # (1, 128, 75)
        audio = encodec_model.decoder(emb)  # (1, 1, ~24000)
        audio = audio.squeeze().cpu().numpy()
    # Truncate/pad to exactly 1 second
    if len(audio) >= ENCODEC_SFREQ:
        audio = audio[:ENCODEC_SFREQ]
    else:
        audio = np.pad(audio, (0, ENCODEC_SFREQ - len(audio)))
    return audio


def audio_to_wav_bytes(audio: np.ndarray, sr: int = ENCODEC_SFREQ) -> bytes:
    """Convert audio numpy array to WAV bytes for st.audio()."""
    buf = BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def get_original_audio_segment(song_id: int, window_start_sec: float) -> bytes | None:
    """Get 1-second original audio segment at 24 kHz as WAV bytes."""
    audio = load_song_audio(song_id)
    start = int(window_start_sec * ENCODEC_SFREQ)
    end = start + ENCODEC_SFREQ
    if end > len(audio):
        return None
    return audio_to_wav_bytes(audio[start:end])


# ---------------------------------------------------------------------------
# Checkpoint discovery (same pattern as viz/reconstruction.py)
# ---------------------------------------------------------------------------


def find_checkpoint_runs() -> list[str]:
    runs = []
    if CHECKPOINTS_DIR.exists():
        for run_dir in CHECKPOINTS_DIR.iterdir():
            if run_dir.is_dir():
                has_checkpoints = any(
                    (d / "config.json").exists()
                    for d in run_dir.iterdir()
                    if d.is_dir()
                )
                if has_checkpoints:
                    runs.append(run_dir.name)
    return sorted(runs, reverse=True)


def find_epochs_for_run(run_name: str) -> list[str]:
    epochs = []
    run_dir = CHECKPOINTS_DIR / run_name
    if run_dir.exists():
        for d in run_dir.iterdir():
            if d.is_dir() and (d / "config.json").exists():
                epochs.append(d.name)
    return sorted(epochs, key=lambda x: (x != "final", x))


def get_default_epoch(epochs: list[str]) -> int:
    if "final" in epochs:
        return epochs.index("final")
    epoch_nums = []
    for i, ep in enumerate(epochs):
        if ep.startswith("epoch_"):
            try:
                epoch_nums.append((int(ep.split("_")[1]), i))
            except (ValueError, IndexError):
                pass
    return max(epoch_nums, key=lambda x: x[0])[1] if epoch_nums else 0


# ---------------------------------------------------------------------------
# Sidebar — model, split, filters, navigation
# ---------------------------------------------------------------------------

st.sidebar.header("Model")
available_runs = find_checkpoint_runs()
checkpoint_path = None

if available_runs:
    selected_run = st.sidebar.selectbox("Run", available_runs, index=0)
    if selected_run:
        epochs = find_epochs_for_run(selected_run)
        default_idx = get_default_epoch(epochs)
        selected_epoch = st.sidebar.selectbox("Epoch", epochs, index=default_idx)
        if selected_epoch:
            checkpoint_path = str(CHECKPOINTS_DIR / selected_run / selected_epoch)

manual_path = st.sidebar.text_input("Or enter checkpoint path:")
if manual_path:
    checkpoint_path = manual_path

st.sidebar.divider()
split = st.sidebar.radio("Split", ["train", "val"], index=1)

# Load data
eeg_tensors, mask_indices, target_embeds, metadata = load_split_data(split)
channel_names = [STANDARD_CHANNELS[i] for i in mask_indices.tolist()]

# Filters
st.sidebar.subheader("Filters")
song_names = sorted(metadata["song_name"].unique().to_list())
selected_songs = st.sidebar.multiselect("Song", song_names, default=[])
subject_ids = sorted(metadata["subject_id"].unique().to_list())
selected_subjects = st.sidebar.multiselect("Subject", subject_ids, default=[])

filtered = metadata.with_row_index("_idx")
if selected_songs:
    filtered = filtered.filter(pl.col("song_name").is_in(selected_songs))
if selected_subjects:
    filtered = filtered.filter(pl.col("subject_id").is_in(selected_subjects))

total_filtered = len(filtered)
st.sidebar.caption(f"{total_filtered:,} / {len(metadata):,} samples")

if total_filtered == 0:
    st.warning("No samples match the current filters.")
    st.stop()

# Navigation
st.sidebar.subheader("Navigate")
if "sample_pos" not in st.session_state:
    st.session_state.sample_pos = 0

col_n1, col_n2, col_n3 = st.sidebar.columns(3)
with col_n1:
    if st.button("Prev") and st.session_state.sample_pos > 0:
        st.session_state.sample_pos -= 1
        st.rerun()
with col_n2:
    if st.button("Next") and st.session_state.sample_pos < total_filtered - 1:
        st.session_state.sample_pos += 1
        st.rerun()
with col_n3:
    if st.button("Random"):
        st.session_state.sample_pos = random.randint(0, total_filtered - 1)
        st.rerun()

st.sidebar.number_input(
    "Index",
    min_value=0,
    max_value=total_filtered - 1,
    key="sample_pos",
)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if not checkpoint_path:
    st.info("Select a checkpoint to begin.")
    st.stop()

# Load models (cached)
with st.spinner("Loading EEG model..."):
    eeg_model = load_eeg_model(checkpoint_path)
with st.spinner("Loading EnCodec decoder..."):
    encodec_model = load_encodec()

# Current sample
row = filtered.row(st.session_state.sample_pos, named=True)
idx = row["_idx"]
eeg_sample = eeg_tensors[idx]  # (120, 125)
target_embed = target_embeds[idx]  # (128, 75)

# Model inference
with torch.no_grad():
    pred_embed = eeg_model(eeg_sample.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

# ---- Sample info + metrics ----
col_info, col_metrics = st.columns([3, 1])
with col_info:
    st.subheader(f"{row['song_name']}")
    st.caption(f"by {row['artist']}")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Subject", row["subject_id"])
    mc2.metric("Window", f"{row['window_idx']} ({row['window_start_sec']:.0f}s)")
    mc3.metric("Tempo", f"{row['tempo_bpm']} BPM")
    mc4.metric("Familiarity", f"{row['familiarity']}/9")

with col_metrics:
    mse = F.mse_loss(pred_embed, target_embed).item()
    cos_sim = F.cosine_similarity(
        pred_embed.reshape(1, -1), target_embed.reshape(1, -1)
    ).item()
    st.metric("MSE", f"{mse:.4f}")
    st.metric("Cosine Sim", f"{cos_sim:.4f}")

# ---- Audio comparison ----
st.subheader("Audio Comparison")

target_audio = decode_embeddings(encodec_model, target_embed)
pred_audio = decode_embeddings(encodec_model, pred_embed)
original_bytes = get_original_audio_segment(row["song_id"], row["window_start_sec"])

ac1, ac2, ac3 = st.columns(3)
with ac1:
    st.markdown("**Original Audio**")
    st.caption("Source FLAC, 1-second window")
    if original_bytes:
        st.audio(original_bytes, format="audio/wav")
    else:
        st.warning("Audio not available")

with ac2:
    st.markdown("**Target Decoded**")
    st.caption("EnCodec encode → decode roundtrip")
    st.audio(audio_to_wav_bytes(target_audio), format="audio/wav")

with ac3:
    st.markdown("**Predicted Decoded**")
    st.caption("EEG model prediction → EnCodec decode")
    st.audio(audio_to_wav_bytes(pred_audio), format="audio/wav")

# ---- EEG input signal ----
st.subheader("EEG Input")
n_ch_display = st.slider("Channels to display", 1, len(channel_names), min(10, len(channel_names)))

eeg_np = eeg_sample.numpy()
time_eeg = np.arange(eeg_np.shape[1]) / EEG_SFREQ

fig_eeg, axes_eeg = plt.subplots(
    n_ch_display, 1, figsize=(14, n_ch_display * 0.5), sharex=True
)
if n_ch_display == 1:
    axes_eeg = [axes_eeg]
for i in range(n_ch_display):
    axes_eeg[i].plot(time_eeg, eeg_np[i], linewidth=0.5, color="#1f77b4")
    axes_eeg[i].set_ylabel(channel_names[i], fontsize=6, rotation=0, ha="right")
    axes_eeg[i].set_yticks([])
    axes_eeg[i].grid(True, alpha=0.2)
axes_eeg[-1].set_xlabel("Time (s)")
plt.tight_layout()
st.pyplot(fig_eeg)
plt.close()

# ---- Embedding heatmaps ----
st.subheader("Embedding Comparison")

target_np = target_embed.numpy()
pred_np = pred_embed.numpy()
diff_np = pred_np - target_np

vmin = min(target_np.min(), pred_np.min())
vmax = max(target_np.max(), pred_np.max())

fig_emb, (ax_t, ax_p, ax_d) = plt.subplots(1, 3, figsize=(18, 4))

ax_t.imshow(target_np, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
ax_t.set_title("Target")
ax_t.set_xlabel("Frame (75 Hz)")
ax_t.set_ylabel("Dim (128)")

ax_p.imshow(pred_np, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
ax_p.set_title("Predicted")
ax_p.set_xlabel("Frame (75 Hz)")

dmax = max(abs(diff_np.min()), abs(diff_np.max()))
im = ax_d.imshow(diff_np, aspect="auto", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
ax_d.set_title("Difference (Pred − Target)")
ax_d.set_xlabel("Frame (75 Hz)")
plt.colorbar(im, ax=ax_d, shrink=0.8)

plt.tight_layout()
st.pyplot(fig_emb)
plt.close()

# ---- Spectrograms ----
st.subheader("Spectrogram Comparison")


def plot_spectrogram(audio, sr, ax, title):
    ax.specgram(audio, Fs=sr, NFFT=512, noverlap=384, cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq (Hz)")
    ax.set_ylim(0, sr // 2)


fig_spec, (ax_s1, ax_s2, ax_s3) = plt.subplots(1, 3, figsize=(18, 4))

if original_bytes:
    orig_audio = load_song_audio(row["song_id"])
    start = int(row["window_start_sec"] * ENCODEC_SFREQ)
    plot_spectrogram(orig_audio[start : start + ENCODEC_SFREQ], ENCODEC_SFREQ, ax_s1, "Original")
else:
    ax_s1.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax_s1.transAxes)
    ax_s1.set_title("Original")

plot_spectrogram(target_audio, ENCODEC_SFREQ, ax_s2, "Target Decoded")
plot_spectrogram(pred_audio, ENCODEC_SFREQ, ax_s3, "Predicted Decoded")

plt.tight_layout()
st.pyplot(fig_spec)
plt.close()
