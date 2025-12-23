import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pathlib import Path
from safetensors.torch import load_file
import random

st.set_page_config(page_title="DEAP Data Inspector", layout="wide")

st.title("DEAP Data Inspector")
st.write("Visualize DEAP EEG data with emotion label filtering")

# Constants
DEFAULT_PATH = "/kreka/research/willy/side/brain_datasets/deap_processed"
COLOR_SIGNAL = "#1f77b4"

# DEAP 32 EEG channel names (standard 10-20 system order from DEAP documentation)
DEAP_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "F4", "F8", "FC6", "FC2", "C4", "T8",
    "CP6", "CP2", "P4", "P8", "PO4", "O2", "Fz", "Cz",
]


@st.cache_resource
def load_deap_data(data_dir: str, split: str):
    """Load DEAP safetensors and corresponding metadata parquet."""
    path = Path(data_dir)

    safetensors_path = path / f"{split}.safetensors"
    metadata_path = path / f"{split}-metadata.parquet"

    if not safetensors_path.exists():
        raise FileNotFoundError(f"Safetensors not found: {safetensors_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    # Load safetensors - DEAP uses "samples" key
    data = load_file(str(safetensors_path))
    tensors = data["samples"]  # [N, 32, 128]

    # Load metadata
    metadata_df = pl.read_parquet(metadata_path)

    return tensors, metadata_df


# Sidebar for filters
st.sidebar.header("Filters")

# Dataset selection
data_dir = st.sidebar.text_input(
    "Data directory:",
    value=DEFAULT_PATH,
)

split = st.sidebar.selectbox("Split:", ["train", "val"])

# Session state
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0

# Load button
if st.sidebar.button("Load Dataset") or st.session_state.loaded:
    try:
        with st.spinner("Loading data..."):
            tensors, metadata_df = load_deap_data(data_dir, split)

        st.session_state.loaded = True

        num_samples, num_channels, seq_len = tensors.shape
        st.success(f"Loaded {num_samples:,} samples | Shape: ({num_channels}, {seq_len})")

        # Sidebar filters
        st.sidebar.subheader("Label Filters")

        # Binary label filters
        col1, col2 = st.sidebar.columns(2)
        with col1:
            filter_valence_high = st.checkbox("Valence High")
            filter_arousal_high = st.checkbox("Arousal High")
        with col2:
            filter_valence_low = st.checkbox("Valence Low")
            filter_arousal_low = st.checkbox("Arousal Low")

        col3, col4 = st.sidebar.columns(2)
        with col3:
            filter_dominance_high = st.checkbox("Dominance High")
            filter_liking_high = st.checkbox("Liking High")
        with col4:
            filter_dominance_low = st.checkbox("Dominance Low")
            filter_liking_low = st.checkbox("Liking Low")

        # Subject filter
        st.sidebar.subheader("Subject Filter")
        subject_opts = sorted(metadata_df["subject_id"].unique().to_list())
        selected_subjects = st.sidebar.multiselect(
            "Subjects:",
            options=subject_opts,
            default=[],
        )

        # Apply filters
        filtered_df = metadata_df.with_row_index("_orig_idx")

        if filter_valence_high:
            filtered_df = filtered_df.filter(pl.col("valence_high") == True)
        if filter_valence_low:
            filtered_df = filtered_df.filter(pl.col("valence_high") == False)
        if filter_arousal_high:
            filtered_df = filtered_df.filter(pl.col("arousal_high") == True)
        if filter_arousal_low:
            filtered_df = filtered_df.filter(pl.col("arousal_high") == False)
        if filter_dominance_high:
            filtered_df = filtered_df.filter(pl.col("dominance_high") == True)
        if filter_dominance_low:
            filtered_df = filtered_df.filter(pl.col("dominance_high") == False)
        if filter_liking_high:
            filtered_df = filtered_df.filter(pl.col("liking_high") == True)
        if filter_liking_low:
            filtered_df = filtered_df.filter(pl.col("liking_high") == False)
        if selected_subjects:
            filtered_df = filtered_df.filter(pl.col("subject_id").is_in(selected_subjects))

        total_filtered = len(filtered_df)
        st.info(f"Showing {total_filtered:,} samples after filtering")

        if total_filtered == 0:
            st.warning("No samples match the current filters")
        else:
            # Sample selection
            st.subheader("Sample Selection")
            col1, col2 = st.columns([3, 1])

            with col1:
                if st.session_state.sample_idx >= total_filtered:
                    st.session_state.sample_idx = 0

                sample_idx = st.number_input(
                    f"Sample index (0 to {total_filtered - 1}):",
                    min_value=0,
                    max_value=total_filtered - 1,
                    value=st.session_state.sample_idx,
                    key="sample_input",
                )
                st.session_state.sample_idx = sample_idx

            with col2:
                st.write("")
                st.write("")
                if st.button("Random Sample"):
                    st.session_state.sample_idx = random.randint(0, total_filtered - 1)
                    st.rerun()

            # Get actual tensor index from filtered dataframe
            row = filtered_df.row(sample_idx, named=True)
            actual_idx = row["_orig_idx"]

            # Layout: metadata | visualization
            col_meta, col_viz = st.columns([1, 3])

            with col_meta:
                st.subheader("Sample Info")

                # Key metadata
                st.metric("Subject", row["subject_id"])
                st.metric("Trial", row["trial_id"])
                st.metric("Window", row["window_id"])

                st.divider()
                st.subheader("Emotion Labels")

                # Continuous labels with color coding
                val_color = "normal" if row["valence"] >= 5 else "off"
                aro_color = "normal" if row["arousal"] >= 5 else "off"
                dom_color = "normal" if row["dominance"] >= 5 else "off"
                lik_color = "normal" if row["liking"] >= 5 else "off"

                st.metric("Valence", f"{row['valence']:.2f}", delta="High" if row["valence_high"] else "Low")
                st.metric("Arousal", f"{row['arousal']:.2f}", delta="High" if row["arousal_high"] else "Low")
                st.metric("Dominance", f"{row['dominance']:.2f}", delta="High" if row["dominance_high"] else "Low")
                st.metric("Liking", f"{row['liking']:.2f}", delta="High" if row["liking_high"] else "Low")

            with col_viz:
                st.subheader("EEG Signal (32 channels, 128 samples)")

                # Get sample data
                sample = tensors[actual_idx].numpy()  # [32, 128]

                # Create figure
                fig, axes = plt.subplots(
                    num_channels, 1,
                    figsize=(14, num_channels * 0.5),
                    sharex=True,
                )

                time = np.arange(seq_len)

                for i, (ax, ch_name) in enumerate(zip(axes, DEAP_CHANNELS)):
                    ax.plot(time, sample[i], linewidth=0.5, color=COLOR_SIGNAL)
                    ax.set_ylabel(ch_name, fontsize=7, rotation=0, ha="right")
                    ax.tick_params(axis="y", labelsize=5)
                    ax.set_xlim(0, seq_len)
                    ax.set_yticks([])

                axes[-1].set_xlabel("Sample")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info("Click 'Load Dataset' in the sidebar to get started.")
