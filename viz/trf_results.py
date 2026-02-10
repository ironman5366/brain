"""
TRF Analysis Results — Interactive Streamlit dashboard.

Loads precomputed TRF results from reports/trf/results.npz and provides
interactive exploration of backward/forward TRF analyses.

Usage:
    uv run streamlit run viz/trf_results.py
"""

import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import soundfile as sf
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from data.nmed.songs import NMED_AUDIO_DIR, SONG_BY_ID

st.set_page_config(page_title="TRF Analysis", layout="wide")
st.title("Temporal Response Function (TRF) Analysis")
st.caption("Does the EEG contain decodable auditory information?")

RESULTS_PATH = Path("reports/trf/results.npz")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_results():
    """Load precomputed TRF results."""
    if not RESULTS_PATH.exists():
        return None
    data = np.load(str(RESULTS_PATH), allow_pickle=True)
    return dict(data)


@st.cache_data
def load_audio_segment(song_id: int, start_sec: float, duration_sec: float) -> bytes | None:
    """Load an audio segment and return as WAV bytes for st.audio."""
    audio_path = NMED_AUDIO_DIR / f"song_{song_id:02d}.flac"
    if not audio_path.exists():
        return None
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    segment = audio[start_sample:min(end_sample, len(audio))]
    buf = BytesIO()
    sf.write(buf, segment, sr, format="WAV")
    buf.seek(0)
    return buf.read()


@st.cache_resource
def get_montage_info():
    """Create MNE Info object with the 32-channel layout for topomap plotting."""
    montage = mne.channels.make_standard_montage("standard_1005")
    info = mne.create_info(
        ch_names=[
            "Fp1", "F3", "F7", "FC5", "FC1", "FCz", "C3", "T7",
            "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz",
            "O2", "P4", "P8", "TP10", "CP6", "CP2", "CPz", "Cz",
            "C4", "T8", "FC6", "FC2", "F4", "F8", "AFz", "Fp2",
        ],
        sfreq=125,
        ch_types="eeg",
    )
    info.set_montage(montage)
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

results = load_results()

if results is None:
    st.error(f"Results file not found at `{RESULTS_PATH}`. "
             "Run `uv run python analysis/trf_analysis.py` first.")
    st.stop()

subject_ids = list(results["subject_ids"])
song_ids = list(results["song_ids"])
channel_names = list(results["channel_names"])
backward_r = results["backward_r"]  # (n_subjects, n_songs)
alpha = float(results["alpha"])

has_forward = "forward_r" in results
has_null = "null_r" in results

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

st.header("Summary")

subject_mean_r = backward_r.mean(axis=1)
grand_mean_r = subject_mean_r.mean()
grand_se = subject_mean_r.std() / np.sqrt(len(subject_ids))

# Compute significance if null available
n_significant = 0
p_values = np.ones(len(subject_ids))
if has_null:
    null_r = results["null_r"]  # (n_subjects, n_perms)
    for i in range(len(subject_ids)):
        p_values[i] = (1 + np.sum(null_r[i] >= subject_mean_r[i])) / (1 + null_r.shape[1])
    n_significant = int(np.sum(p_values < 0.05))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Grand Mean r", f"{grand_mean_r:.4f}")
col2.metric("SE", f"{grand_se:.4f}")
if has_null:
    col3.metric("Significant Subjects", f"{n_significant}/{len(subject_ids)}")
    col4.metric("Null Mean r", f"{null_r.mean():.4f}")
else:
    col3.metric("Subjects", str(len(subject_ids)))
    col4.metric("Songs", str(len(song_ids)))

if grand_mean_r > 0.02:
    st.success(f"Signal DETECTED. Grand mean r = {grand_mean_r:.4f} "
               f"(literature reference: music envelope tracking ~ 0.05-0.15)")
else:
    st.warning(f"Signal NOT detected. Grand mean r = {grand_mean_r:.4f} "
               f"(below typical detection threshold of ~0.02)")

st.markdown(f"**Ridge alpha:** {alpha:.0e} | "
            f"**Lags:** {results['lags_backward_ms'][0]:.0f}ms to "
            f"{results['lags_backward_ms'][-1]:.0f}ms")

# ---------------------------------------------------------------------------
# Per-subject bar chart
# ---------------------------------------------------------------------------

st.header("Backward TRF: Per-Subject Reconstruction Correlation")

fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(len(subject_ids))
colors = []
for i in range(len(subject_ids)):
    if has_null and p_values[i] < 0.05:
        colors.append("#2ca02c")  # green = significant
    elif has_null and p_values[i] < 0.1:
        colors.append("#ff7f0e")  # orange = marginal
    else:
        colors.append("#1f77b4")  # blue = not significant

ax.bar(x, subject_mean_r, color=colors, edgecolor="white", linewidth=0.5)

if has_null:
    # 95th percentile of null as threshold line
    null_95 = np.percentile(null_r.mean(axis=0), 95)
    ax.axhline(null_95, color="red", linestyle="--", linewidth=1, label=f"Null 95th pct ({null_95:.4f})")
    ax.legend()

ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(subject_ids, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean Pearson r")
ax.set_title("Backward TRF: Audio Envelope Reconstruction from EEG")
plt.tight_layout()
st.pyplot(fig)
plt.close()

if has_null:
    st.caption("Green = p < 0.05, Orange = p < 0.10, Blue = not significant")

# ---------------------------------------------------------------------------
# Per-song heatmap
# ---------------------------------------------------------------------------

st.header("Subject x Song Correlation Matrix")

fig, ax = plt.subplots(figsize=(10, 8))
song_labels = [f"{sid}: {SONG_BY_ID[sid].title[:20]}" if sid in SONG_BY_ID else str(sid)
               for sid in song_ids]
im = ax.imshow(backward_r, cmap="RdBu_r", vmin=-0.1, vmax=0.2, aspect="auto")
ax.set_xticks(range(len(song_ids)))
ax.set_xticklabels(song_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(subject_ids)))
ax.set_yticklabels(subject_ids, fontsize=8)
ax.set_xlabel("Song")
ax.set_ylabel("Subject")
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Backward TRF Correlation: Subject x Song")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ---------------------------------------------------------------------------
# Forward TRF viewer
# ---------------------------------------------------------------------------

if has_forward:
    st.header("Forward TRF (Encoding Model)")

    forward_r = results["forward_r"]  # (n_subjects, n_songs, 32)
    forward_weights = results["forward_weights"]  # (n_subjects, 32, n_lags)
    lags_forward_ms = results["lags_forward_ms"]

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("TRF Weights")
        show_grand_avg = st.checkbox("Grand average (all subjects)", value=True)
        if show_grand_avg:
            weights_to_plot = forward_weights.mean(axis=0)  # (32, n_lags)
            title = "Grand Average Forward TRF"
        else:
            subj_choice = st.selectbox("Subject", subject_ids, key="fwd_subj")
            s_idx = subject_ids.index(subj_choice)
            weights_to_plot = forward_weights[s_idx]  # (32, n_lags)
            title = f"Forward TRF: {subj_choice}"

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            weights_to_plot,
            aspect="auto",
            cmap="RdBu_r",
            extent=[lags_forward_ms[0], lags_forward_ms[-1], len(channel_names) - 0.5, -0.5],
        )
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Channel")
        ax.set_yticks(range(len(channel_names)))
        ax.set_yticklabels(channel_names, fontsize=7)
        plt.colorbar(im, ax=ax, label="Weight")
        ax.set_title(title)
        # Mark key latencies
        for lag_ms, label in [(80, "P1"), (100, "N1"), (200, "P2")]:
            ax.axvline(lag_ms, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.text(lag_ms + 5, -0.3, label, fontsize=8, color="gray")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Topomap at Selected Latency")
        lag_slider = st.slider(
            "Latency (ms)",
            min_value=int(lags_forward_ms[0]),
            max_value=int(lags_forward_ms[-1]),
            value=100,
            step=8,
        )
        # Find nearest lag index
        lag_idx = np.argmin(np.abs(lags_forward_ms - lag_slider))
        actual_lag_ms = lags_forward_ms[lag_idx]

        grand_weights = forward_weights.mean(axis=0)  # (32, n_lags)
        weights_at_lag = grand_weights[:, lag_idx]

        info = get_montage_info()
        fig, ax = plt.subplots(figsize=(5, 5))
        mne.viz.plot_topomap(
            weights_at_lag, info, axes=ax, show=False,
            cmap="RdBu_r", vlim=(-np.abs(weights_at_lag).max(), np.abs(weights_at_lag).max()),
        )
        ax.set_title(f"Forward TRF at {actual_lag_ms:.0f}ms")
        st.pyplot(fig)
        plt.close()

        # Per-channel encoding correlation
        st.subheader("Best Encoding Channels")
        mean_r_per_ch = forward_r.mean(axis=(0, 1))  # (32,)
        sorted_idx = np.argsort(mean_r_per_ch)[::-1]
        for rank, ch_idx in enumerate(sorted_idx[:10]):
            st.text(f"  {rank+1}. {channel_names[ch_idx]}: r = {mean_r_per_ch[ch_idx]:.4f}")

# ---------------------------------------------------------------------------
# Reconstruction example
# ---------------------------------------------------------------------------

st.header("Reconstruction Example")
st.caption("Listen to the audio while viewing predicted vs actual envelope")

col_s, col_song, col_seg = st.columns(3)
with col_s:
    recon_subject = st.selectbox("Subject", subject_ids, key="recon_subj")
with col_song:
    song_options = {sid: f"{sid}: {SONG_BY_ID[sid].title}" if sid in SONG_BY_ID else str(sid)
                    for sid in song_ids}
    recon_song = st.selectbox("Song", list(song_options.keys()),
                              format_func=lambda x: song_options[x], key="recon_song")
with col_seg:
    segment_start = st.number_input("Start (sec)", min_value=0, max_value=250,
                                    value=30, step=10, key="seg_start")

# Show the per-song correlation for this pair
s_idx = subject_ids.index(recon_subject)
song_idx = song_ids.index(recon_song)
r_val = backward_r[s_idx, song_idx]
st.metric(f"Reconstruction r for {recon_subject}, Song {recon_song}", f"{r_val:.4f}")

# Audio playback
audio_bytes = load_audio_segment(int(recon_song), float(segment_start), 30.0)
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
