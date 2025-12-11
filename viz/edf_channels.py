import streamlit as st
import mne
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="EDF Viewer", layout="wide")

st.title("EDF File Viewer")
st.write("Visualize EEG/biosignal channels from EDF files using MNE")

# File path input
edf_path = st.text_input(
    "Enter the path to your EDF file:", placeholder="/path/to/your/file.edf"
)

if edf_path:
    try:
        with st.spinner("Loading EDF file..."):
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        st.success(f"Successfully loaded: {edf_path}")

        # Display file info
        st.subheader("File Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Channels", len(raw.ch_names))
        with col2:
            st.metric("Sampling Rate", f"{raw.info['sfreq']} Hz")
        with col3:
            duration = raw.n_times / raw.info["sfreq"]
            st.metric("Duration", f"{duration:.2f} seconds")

        # Channel list
        with st.expander("Channel Names"):
            st.write(", ".join(raw.ch_names))

        # Visualization options
        st.subheader("Visualization Settings")
        col1, col2 = st.columns(2)

        with col1:
            selected_channels = st.multiselect(
                "Select channels to display:",
                options=raw.ch_names,
                default=raw.ch_names[: min(5, len(raw.ch_names))],
            )

        with col2:
            start_time = st.number_input(
                "Start time (seconds):",
                min_value=0.0,
                max_value=float(duration),
                value=0.0,
            )
            window_duration = st.number_input(
                "Window duration (seconds):",
                min_value=1.0,
                max_value=float(duration),
                value=min(10.0, duration),
            )

        if selected_channels:
            st.subheader("Channel Visualization")

            # Get data for selected channels
            picks = mne.pick_channels(raw.ch_names, selected_channels)
            start_sample = int(start_time * raw.info["sfreq"])
            end_sample = int((start_time + window_duration) * raw.info["sfreq"])
            end_sample = min(end_sample, raw.n_times)

            data, times = raw[picks, start_sample:end_sample]
            times = times - times[0] + start_time

            # Create figure
            fig, axes = plt.subplots(
                len(selected_channels),
                1,
                figsize=(12, 2 * len(selected_channels)),
                sharex=True,
            )
            if len(selected_channels) == 1:
                axes = [axes]

            for idx, (ax, ch_name) in enumerate(zip(axes, selected_channels)):
                ax.plot(times, data[idx] * 1e6, linewidth=0.5, color="#1f77b4")
                ax.set_ylabel(f"{ch_name}\n(μV)")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(times[0], times[-1])

            axes[-1].set_xlabel("Time (seconds)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Power spectral density
            st.subheader("Power Spectral Density")

            fig_psd, ax_psd = plt.subplots(figsize=(12, 4))
            raw_subset = raw.copy().pick_channels(selected_channels)
            spectrum = raw_subset.compute_psd(fmax=100, verbose=False)
            spectrum.plot(axes=ax_psd, show=False)
            ax_psd.set_title("Power Spectral Density")
            st.pyplot(fig_psd)
            plt.close()
        else:
            st.warning("Please select at least one channel to visualize.")

    except FileNotFoundError:
        st.error(f"File not found: {edf_path}")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
else:
    st.info("Please enter the path to an EDF file to get started.")
