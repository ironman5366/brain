import sys
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
import polars as pl
from pathlib import Path
from safetensors.torch import load_file
import random

sys.path.append(str(Path(__file__).parent.parent))

from constants import STANDARD_CHANNELS

st.set_page_config(page_title="EEG Data Inspector", layout="wide")

st.title("EEG Data Inspector")
st.write("Visualize epoch-based EEG data with metadata from safetensors files")

# Constants
DEFAULT_PATH = "/kreka/research/willy/side/brain_datasets/alljoined-epochs-2025-12-21/alljoined-epochs-2025-12-21-val.safetensors"
COLOR_SIGNAL = "#1f77b4"


@st.cache_resource
def load_data(safetensors_path: str):
    """Load safetensors and corresponding metadata parquet."""
    path = Path(safetensors_path)

    # Load safetensors
    data = load_file(str(path))
    tensors = data["sparse_samples"]  # [N, C, S]
    mask_indices = data["mask_indices"]  # [C_active]

    # Derive metadata path
    metadata_path = path.parent / f"{path.stem}-metadata.parquet"
    if not metadata_path.exists():
        # Try alternative naming
        metadata_path = Path(str(path).replace(".safetensors", "-metadata.parquet"))

    if metadata_path.exists():
        metadata_df = pl.read_parquet(metadata_path)
    else:
        st.warning(f"Metadata not found at {metadata_path}")
        metadata_df = None

    return tensors, mask_indices, metadata_df


def get_channel_names(mask_indices: torch.Tensor) -> list[str]:
    """Get channel names for active channels."""
    return [STANDARD_CHANNELS[i] for i in mask_indices.tolist()]


# File path input
st.subheader("Dataset Selection")
dataset_path = st.text_input(
    "Safetensors file path:",
    value=DEFAULT_PATH,
    placeholder=DEFAULT_PATH,
)

# Session state initialization
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0

# Load button
if dataset_path:
    if st.button("Load Dataset") or st.session_state.loaded:
        try:
            with st.spinner("Loading data..."):
                tensors, mask_indices, metadata_df = load_data(dataset_path)

            st.session_state.loaded = True
            st.success(f"Loaded {len(tensors)} samples")

            # Get channel names
            channel_names = get_channel_names(mask_indices)
            num_channels = len(channel_names)
            seq_len = tensors.shape[2]

            st.caption(f"Shape: {tensors.shape} | Channels: {num_channels} | Sequence length: {seq_len}")

            # Filters
            if metadata_df is not None:
                with st.expander("Filters", expanded=False):
                    col1, col2, col3 = st.columns(3)

                    # Get unique values for filters (handle None values)
                    partition_opts = sorted(
                        [v for v in metadata_df["partition"].unique().to_list() if v is not None]
                    )
                    subject_opts = sorted(
                        [v for v in metadata_df["subject"].unique().to_list() if v is not None]
                    )

                    # Super category might not exist in all datasets
                    super_cat_opts = []
                    if "super_category" in metadata_df.columns:
                        super_cat_opts = sorted(
                            [v for v in metadata_df["super_category"].unique().to_list() if v is not None]
                        )

                    with col1:
                        selected_partitions = st.multiselect(
                            "Partition",
                            options=partition_opts,
                            default=[],
                        )
                    with col2:
                        selected_categories = st.multiselect(
                            "Super Category",
                            options=super_cat_opts,
                            default=[],
                        )
                    with col3:
                        selected_subjects = st.multiselect(
                            "Subject",
                            options=subject_opts,
                            default=[],
                        )

                # Apply filters
                filtered_df = metadata_df
                if selected_partitions:
                    filtered_df = filtered_df.filter(pl.col("partition").is_in(selected_partitions))
                if selected_categories and "super_category" in filtered_df.columns:
                    filtered_df = filtered_df.filter(pl.col("super_category").is_in(selected_categories))
                if selected_subjects:
                    filtered_df = filtered_df.filter(pl.col("subject").is_in(selected_subjects))

                # Get valid indices (row indices in filtered df map to tensor indices)
                valid_indices = list(range(len(filtered_df)))
                total_filtered = len(filtered_df)
            else:
                valid_indices = list(range(len(tensors)))
                total_filtered = len(tensors)
                filtered_df = None

            st.info(f"Showing {total_filtered} samples after filtering")

            if total_filtered == 0:
                st.warning("No samples match the current filters")
            else:
                # Sample selection
                st.subheader("Sample Selection")
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    # Clamp sample_idx to valid range
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
                    st.write("")  # Spacing
                    st.write("")
                    if st.button("Random Sample"):
                        st.session_state.sample_idx = random.randint(0, total_filtered - 1)
                        st.rerun()

                # Get actual tensor index
                if filtered_df is not None and metadata_df is not None:
                    # Map filtered index back to original index
                    # The filtered_df maintains row order, so we need the original row index
                    actual_idx = filtered_df.row(sample_idx, named=True).get("epoch_idx", sample_idx)
                    if actual_idx is None:
                        actual_idx = sample_idx  # Fallback
                else:
                    actual_idx = sample_idx

                # Display
                col_meta, col_viz = st.columns([1, 3])

                with col_meta:
                    st.subheader("Metadata")
                    if filtered_df is not None:
                        row = filtered_df.row(sample_idx, named=True)
                        # Display key metadata fields
                        display_fields = [
                            "image_id", "partition", "category_name", "super_category",
                            "subject", "session", "block", "onset", "seq_num",
                            "category_num", "super_category_id", "epoch_idx"
                        ]
                        for field in display_fields:
                            if field in row and row[field] is not None:
                                st.metric(field, row[field])

                        # Show all other fields in expander
                        with st.expander("All fields"):
                            for k, v in row.items():
                                if k not in display_fields and v is not None:
                                    st.text(f"{k}: {v}")
                    else:
                        st.write(f"Sample index: {actual_idx}")

                with col_viz:
                    st.subheader("EEG Signal")

                    # Get sample data
                    sample = tensors[actual_idx].numpy()  # [C, S]

                    # Create figure
                    fig, axes = plt.subplots(
                        num_channels, 1,
                        figsize=(14, num_channels * 0.6),
                        sharex=True,
                    )
                    if num_channels == 1:
                        axes = [axes]

                    time = np.arange(seq_len)

                    for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
                        ax.plot(time, sample[i], linewidth=0.5, color=COLOR_SIGNAL)
                        ax.set_ylabel(ch_name, fontsize=8, rotation=0, ha="right")
                        ax.tick_params(axis="y", labelsize=6)
                        ax.set_xlim(0, seq_len)
                        ax.grid(True, alpha=0.2)

                    axes[-1].set_xlabel("Sample")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        except FileNotFoundError:
            st.error(f"File not found: {dataset_path}")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
else:
    st.info("Please enter the path to a safetensors file to get started.")
