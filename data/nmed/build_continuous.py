"""
Build continuous (multi-second) NMED-T dataset from existing 1-second processed data.

Groups consecutive 1-second EEG windows + EnCodec embeddings into multi-second sequences.
Reuses the existing processed safetensors and metadata (avoids reprocessing raw .mat files).

Usage:
    uv run python -m data.nmed.build_continuous --window-seconds 2
    uv run python -m data.nmed.build_continuous --window-seconds 5
    uv run python -m data.nmed.build_continuous --window-seconds 10
    uv run python -m data.nmed.build_continuous --window-seconds 30
    uv run python -m data.nmed.build_continuous --window-seconds -1   # full song
"""

import argparse
from pathlib import Path

import polars as pl
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from data.nmed.songs import NMED_OUTPUT_DIR


CONTINUOUS_DIR = NMED_OUTPUT_DIR / "continuous"


def build_continuous(window_seconds: int):
    """Group existing 1-second windows into multi-second sequences."""
    full_song = window_seconds == -1
    label = "full" if full_song else f"w{window_seconds}"
    out_dir = CONTINUOUS_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Building continuous data: {label} ===")

    for split in ["train", "val"]:
        print(f"\n[{split}]")

        # Load existing 1-second data
        eeg_path = NMED_OUTPUT_DIR / f"nmed-{split}.safetensors"
        encodec_path = NMED_OUTPUT_DIR / f"nmed-{split}-encodec.safetensors"
        meta_path = NMED_OUTPUT_DIR / f"nmed-{split}-metadata.parquet"

        print(f"  Loading from {eeg_path}...")
        eeg_data = load_file(eeg_path)
        eeg_samples = eeg_data["sparse_samples"]  # (N, 120, 125)
        mask_indices = eeg_data["mask_indices"]

        print(f"  Loading from {encodec_path}...")
        audio_embeds = load_file(encodec_path)["audio_embeds"]  # (N, 128, 75)

        meta = pl.read_parquet(meta_path)

        print(f"  EEG: {eeg_samples.shape}, Audio: {audio_embeds.shape}")
        print(f"  Metadata: {meta.shape}")

        # Group by (subject_id, song_id) and create sequences
        sequences_eeg = []
        sequences_audio = []
        sequences_meta = []
        seq_lengths = []

        # Get unique (subject, song) pairs
        pairs = (
            meta.select("subject_id", "song_id")
            .unique()
            .sort("subject_id", "song_id")
        )

        for row in tqdm(
            pairs.iter_rows(named=True),
            total=len(pairs),
            desc=f"  Building {label} sequences",
        ):
            subj = row["subject_id"]
            song = row["song_id"]

            # Get indices for this subject+song, sorted by window_idx
            mask = (meta["subject_id"] == subj) & (meta["song_id"] == song)
            indices = meta.with_row_index("idx").filter(mask).sort("window_idx")
            idx_list = indices["idx"].to_list()
            n_windows = len(idx_list)

            if full_song:
                # One sequence per (subject, song) — the entire recording
                seq_eeg = eeg_samples[idx_list]  # (n_windows, 120, 125)
                seq_audio = audio_embeds[idx_list]  # (n_windows, 128, 75)
                sequences_eeg.append(seq_eeg)
                sequences_audio.append(seq_audio)
                seq_lengths.append(n_windows)
                sequences_meta.append({
                    "subject_id": subj,
                    "song_id": song,
                    "song_name": indices["song_name"][0],
                    "artist": indices["artist"][0],
                    "sequence_idx": 0,
                    "start_sec": 0.0,
                    "n_seconds": n_windows,
                })
            else:
                # Non-overlapping windows of size window_seconds
                n_sequences = n_windows // window_seconds
                for seq_idx in range(n_sequences):
                    start = seq_idx * window_seconds
                    end = start + window_seconds
                    chunk_indices = idx_list[start:end]

                    seq_eeg = eeg_samples[chunk_indices]  # (W, 120, 125)
                    seq_audio = audio_embeds[chunk_indices]  # (W, 128, 75)
                    sequences_eeg.append(seq_eeg)
                    sequences_audio.append(seq_audio)
                    sequences_meta.append({
                        "subject_id": subj,
                        "song_id": song,
                        "song_name": indices["song_name"][0],
                        "artist": indices["artist"][0],
                        "sequence_idx": seq_idx,
                        "start_sec": float(start),
                        "n_seconds": window_seconds,
                    })

        print(f"  Created {len(sequences_eeg)} sequences")

        if full_song:
            # Pad to max length
            max_len = max(seq_lengths)
            C, T = eeg_samples.shape[1], eeg_samples.shape[2]
            A_dim, A_frames = audio_embeds.shape[1], audio_embeds.shape[2]

            padded_eeg = torch.zeros(len(sequences_eeg), max_len, C, T)
            padded_audio = torch.zeros(len(sequences_audio), max_len, A_dim, A_frames)

            for i, (seq_e, seq_a, length) in enumerate(
                zip(sequences_eeg, sequences_audio, seq_lengths)
            ):
                padded_eeg[i, :length] = seq_e
                padded_audio[i, :length] = seq_a

            lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long)

            save_file(
                {"sparse_samples": padded_eeg, "mask_indices": mask_indices},
                out_dir / f"nmed-{split}.safetensors",
            )
            save_file(
                {"audio_embeds": padded_audio},
                out_dir / f"nmed-{split}-encodec.safetensors",
            )
            save_file(
                {"lengths": lengths_tensor},
                out_dir / f"nmed-{split}-lengths.safetensors",
            )

            print(f"  Padded EEG: {padded_eeg.shape}")
            print(f"  Padded Audio: {padded_audio.shape}")
            print(f"  Lengths: min={min(seq_lengths)}, max={max_len}, mean={sum(seq_lengths)/len(seq_lengths):.0f}")
        else:
            # Stack into single tensor (all same length)
            stacked_eeg = torch.stack(sequences_eeg)  # (N, W, 120, 125)
            stacked_audio = torch.stack(sequences_audio)  # (N, W, 128, 75)

            save_file(
                {"sparse_samples": stacked_eeg, "mask_indices": mask_indices},
                out_dir / f"nmed-{split}.safetensors",
            )
            save_file(
                {"audio_embeds": stacked_audio},
                out_dir / f"nmed-{split}-encodec.safetensors",
            )

            print(f"  Stacked EEG: {stacked_eeg.shape}")
            print(f"  Stacked Audio: {stacked_audio.shape}")

        # Save metadata
        meta_df = pl.from_dicts(sequences_meta)
        meta_df.write_parquet(out_dir / f"nmed-{split}-metadata.parquet")
        print(f"  Metadata: {meta_df.shape}")

    print(f"\nDone! Output in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build continuous NMED data")
    parser.add_argument(
        "--window-seconds",
        type=int,
        required=True,
        help="Window size in seconds, or -1 for full song",
    )
    args = parser.parse_args()
    build_continuous(args.window_seconds)


if __name__ == "__main__":
    main()
