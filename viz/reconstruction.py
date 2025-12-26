import sys
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import random

sys.path.append(str(Path.cwd().parent))
print("appended", str(Path.cwd().parent))

from models.et import EEGMAE
from data.dataset import MaskedEEGDataset, SparseDataset

st.set_page_config(page_title="MAE Reconstruction Viewer", layout="wide")

st.title("MAE Reconstruction Viewer")
st.write("Visualize MAE checkpoint reconstructions on EEG data")

# Constants
DEFAULT_VAL_PATH = "/kreka/research/willy/side/brain_datasets/alljoined-window-2025-12-14/alljoined-window-2025-12-14-val.safetensors"
CHECKPOINTS_DIR = Path().cwd().parent / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Colors
COLOR_ORIGINAL = "#1f77b4"
COLOR_MASKED = "#888888"
COLOR_RECONSTRUCTED = "#ff7f0e"


def find_checkpoint_runs():
    """Scan checkpoints directory for available checkpoint runs."""
    runs = []
    if CHECKPOINTS_DIR.exists():
        for run_dir in CHECKPOINTS_DIR.iterdir():
            if run_dir.is_dir():
                # Check if this run has any valid checkpoints
                has_checkpoints = any(
                    (d / "config.json").exists()
                    for d in run_dir.iterdir()
                    if d.is_dir()
                )
                if has_checkpoints:
                    runs.append(run_dir.name)
    return sorted(runs, reverse=True)


def find_epochs_for_run(run_name: str):
    """Find all epochs for a given checkpoint run."""
    run_dir = CHECKPOINTS_DIR / run_name
    epochs = []
    if run_dir.exists():
        for checkpoint_dir in run_dir.iterdir():
            if checkpoint_dir.is_dir() and (checkpoint_dir / "config.json").exists():
                epochs.append(checkpoint_dir.name)
    return epochs


def get_default_epoch(epochs: list[str]) -> int:
    """Get the default epoch index. Prefer 'final', otherwise highest epoch number."""
    if "final" in epochs:
        return epochs.index("final")

    # Try to find highest epoch number
    epoch_nums = []
    for i, ep in enumerate(epochs):
        if ep.startswith("epoch_"):
            try:
                num = int(ep.split("_")[1])
                epoch_nums.append((num, i))
            except (ValueError, IndexError):
                pass

    if epoch_nums:
        # Return index of highest epoch number
        return max(epoch_nums, key=lambda x: x[0])[1]

    return 0  # Default to first


@st.cache_resource
def load_model(checkpoint_path: str):
    """Load MAE model from checkpoint."""
    return EEGMAE.from_pretrained(checkpoint_path).to(DEVICE).eval()


@st.cache_resource
def load_dataset(dataset_path: str):
    """Load EEG dataset. Returns sparse dataset for model, but keeps mask for visualization."""
    # Use EEGDataset to get the mask, but we'll use sparse data for model inference
    dense_ds = MaskedEEGDataset(Path(dataset_path))
    sparse_ds = SparseDataset(Path(dataset_path))
    return sparse_ds, dense_ds.mask


# Checkpoint selection
st.subheader("Model Selection")
available_runs = find_checkpoint_runs()

checkpoint_path = None
if available_runs:
    col1, col2 = st.columns(2)
    with col1:
        selected_run = st.selectbox(
            "Checkpoint run:",
            options=available_runs,
            index=0,
        )
    with col2:
        if selected_run:
            epochs = find_epochs_for_run(selected_run)
            default_idx = get_default_epoch(epochs)
            selected_epoch = st.selectbox(
                "Epoch:",
                options=epochs,
                index=default_idx,
            )
            if selected_epoch:
                checkpoint_path = str(CHECKPOINTS_DIR / selected_run / selected_epoch)
else:
    st.warning("No checkpoints found in checkpoints/ directory")

# Dataset selection
st.subheader("Dataset Selection")
dataset_path = st.text_input(
    "Dataset path (.safetensors):",
    value=DEFAULT_VAL_PATH,
    placeholder=DEFAULT_VAL_PATH,
)

# Track if visualization is active
if "viz_active" not in st.session_state:
    st.session_state.viz_active = False

if checkpoint_path and dataset_path:
    if st.button("Load & Visualize"):
        st.session_state.viz_active = True

# Load model and dataset
if st.session_state.viz_active and checkpoint_path and dataset_path:
    try:
        with st.spinner("Loading model..."):
            model = load_model(checkpoint_path)
        st.success(f"Model loaded from: {checkpoint_path}")
        st.caption(
            f"Masking mode: **{model.mask_on}** | Masking ratio: **{model.masking_ratio}**"
        )

        with st.spinner("Loading dataset..."):
            dataset, channel_mask = load_dataset(dataset_path)
        st.success(f"Dataset loaded: {len(dataset)} samples")

        # Sample selection
        st.subheader("Sample Selection")
        col1, col2 = st.columns([3, 1])

        # Initialize sample index in session state
        if "sample_idx" not in st.session_state:
            st.session_state.sample_idx = 0

        with col1:
            sample_idx = st.number_input(
                "Sample index:",
                min_value=0,
                max_value=len(dataset) - 1,
                value=st.session_state.sample_idx,
                key="sample_input",
            )
            st.session_state.sample_idx = sample_idx

        with col2:
            if st.button("Random Sample"):
                st.session_state.sample_idx = random.randint(0, len(dataset) - 1)
                st.rerun()

        # Get sample and run through model
        sample = dataset[sample_idx]  # [num_active_channels, S] sparse representation
        num_active_channels, sample_len = sample.shape

        with torch.inference_mode():
            batch = sample.unsqueeze(0).to(DEVICE)  # [1, C, S]
            result = model(batch, return_debug=True)

        masked_indices = result["masked_indices"][0].cpu().numpy()  # [num_masked]
        unmasked_indices = result["unmasked_indices"][0].cpu().numpy()  # [num_unmasked]
        mask_on = result["mask_on"]
        decoded = (
            result["decoded"][0].cpu().numpy()
        )  # [num_tokens, seq_len] or [C, S] depending on mask_on
        loss = result["loss"].item()

        st.metric("Sample MSE Loss", f"{loss:.6f}")

        # Prepare data for visualization
        original = sample.numpy()  # [num_active_channels, S] - sparse

        # Handle different masking modes
        if mask_on == "samples":
            # decoded is [S, C] after model permutes, need to permute back
            decoded = decoded.T  # [C, S]

        # Get active channel indices from channel mask (for labeling)
        active_channel_indices = np.where(channel_mask.numpy())[0]
        num_active = len(active_channel_indices)

        # Create figure with 3 subplots
        st.subheader("Visualizations")

        # 1. Original
        st.markdown("### Original Signal")
        fig1, axes1 = plt.subplots(
            num_active, 1, figsize=(14, num_active * 0.8), sharex=True
        )
        if num_active == 1:
            axes1 = [axes1]

        time = np.arange(sample_len)
        for i, ch_idx in enumerate(active_channel_indices):
            # original[i] is the i-th active channel (sparse indexing)
            axes1[i].plot(time, original[i], linewidth=0.5, color=COLOR_ORIGINAL)
            axes1[i].set_ylabel(f"Ch {ch_idx}", fontsize=8)
            axes1[i].tick_params(axis="y", labelsize=6)
            axes1[i].set_xlim(0, sample_len)

        axes1[-1].set_xlabel("Sample")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        # 2. Masked Version
        st.markdown("### Masked Version")
        st.caption(
            f"Masking on: **{mask_on}** | Masked tokens: {len(masked_indices)} / {len(masked_indices) + len(unmasked_indices)}"
        )

        fig2, axes2 = plt.subplots(
            num_active, 1, figsize=(14, num_active * 0.8), sharex=True
        )
        if num_active == 1:
            axes2 = [axes2]

        if mask_on == "channels":
            # Masked indices refer to channel positions in the sparse representation
            masked_set = set(masked_indices)
            for i, ch_idx in enumerate(active_channel_indices):
                # original[i] is sparse indexing; masked_indices are also sparse indices
                if i in masked_set:
                    color = COLOR_MASKED
                    alpha = 0.5
                else:
                    color = COLOR_ORIGINAL
                    alpha = 1.0
                axes2[i].plot(
                    time, original[i], linewidth=0.5, color=color, alpha=alpha
                )
                axes2[i].set_ylabel(f"Ch {ch_idx}", fontsize=8)
                axes2[i].tick_params(axis="y", labelsize=6)
                axes2[i].set_xlim(0, sample_len)
        else:
            # mask_on == "samples" - masked time regions
            masked_set = set(masked_indices)
            for i, ch_idx in enumerate(active_channel_indices):
                # Plot original in blue (sparse indexing)
                axes2[i].plot(time, original[i], linewidth=0.5, color=COLOR_ORIGINAL)
                axes2[i].set_ylabel(f"Ch {ch_idx}", fontsize=8)
                axes2[i].tick_params(axis="y", labelsize=6)
                axes2[i].set_xlim(0, sample_len)

                # Highlight masked time samples
                for t in masked_indices:
                    axes2[i].axvline(x=t, color=COLOR_MASKED, alpha=0.3, linewidth=0.5)

        axes2[-1].set_xlabel("Sample")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # 3. Reconstructed Version
        st.markdown("### Reconstruction")
        st.caption("Original in blue, reconstructed patches in orange")

        fig3, axes3 = plt.subplots(
            num_active, 1, figsize=(14, num_active * 0.8), sharex=True
        )
        if num_active == 1:
            axes3 = [axes3]

        if mask_on == "channels":
            masked_set = set(masked_indices)
            for i, ch_idx in enumerate(active_channel_indices):
                if i in masked_set:
                    # Show reconstructed channel
                    axes3[i].plot(
                        time,
                        decoded[i],
                        linewidth=0.5,
                        color=COLOR_RECONSTRUCTED,
                        label="Reconstructed",
                    )
                    # Also show original as dashed for comparison (sparse indexing)
                    axes3[i].plot(
                        time,
                        original[i],
                        linewidth=0.3,
                        color=COLOR_ORIGINAL,
                        alpha=0.5,
                        linestyle="--",
                    )
                else:
                    # Show original (unmasked) - sparse indexing
                    axes3[i].plot(
                        time, original[i], linewidth=0.5, color=COLOR_ORIGINAL
                    )
                axes3[i].set_ylabel(f"Ch {ch_idx}", fontsize=8)
                axes3[i].tick_params(axis="y", labelsize=6)
                axes3[i].set_xlim(0, sample_len)
        else:
            # mask_on == "samples"
            masked_set = set(masked_indices)
            for i, ch_idx in enumerate(active_channel_indices):
                # Plot full original (sparse indexing)
                axes3[i].plot(time, original[i], linewidth=0.5, color=COLOR_ORIGINAL)
                # Overlay reconstructed at masked positions
                recon_vals = np.full(sample_len, np.nan)
                for t in masked_indices:
                    if t < sample_len:
                        recon_vals[t] = decoded[i, t]
                axes3[i].scatter(
                    masked_indices[masked_indices < sample_len],
                    [decoded[i, t] for t in masked_indices if t < sample_len],
                    s=1,
                    color=COLOR_RECONSTRUCTED,
                    alpha=0.8,
                )
                axes3[i].set_ylabel(f"Ch {ch_idx}", fontsize=8)
                axes3[i].tick_params(axis="y", labelsize=6)
                axes3[i].set_xlim(0, sample_len)

        axes3[-1].set_xlabel("Sample")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback

        st.code(traceback.format_exc())
elif checkpoint_path and dataset_path:
    st.info("Click 'Load & Visualize' to start.")
else:
    st.info("Please select a checkpoint and provide a dataset path to get started.")
