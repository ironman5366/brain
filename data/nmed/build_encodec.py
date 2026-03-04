"""
Pre-compute EnCodec encoder embeddings for all NMED-T audio windows.

Produces audio_embeds tensors aligned with the EEG sample ordering
from the processed dataset.

Usage:
    uv run python -m data.nmed.build_encodec
"""

from pathlib import Path

import polars as pl
import soundfile as sf
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import EncodecModel

from data.nmed.songs import (
    NMED_AUDIO_DIR,
    NMED_OUTPUT_DIR,
    NMED_SFREQ,
    SONG_BY_ID,
)

ENCODEC_SFREQ = 24000
ENCODEC_HOP = 320  # downsampling factor: 24000/320 = 75 frames/sec
FRAMES_PER_SECOND = ENCODEC_SFREQ // ENCODEC_HOP  # 75
WINDOW_SECONDS = 1.0


def load_audio_24khz(song_id: int) -> torch.Tensor:
    """Load a song's audio and resample to 24kHz mono."""
    path = NMED_AUDIO_DIR / f"song_{song_id:02d}.flac"
    audio, sr = sf.read(path)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio_tensor = torch.from_numpy(audio).float()

    # Resample to 24kHz if needed
    if sr != ENCODEC_SFREQ:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(sr, ENCODEC_SFREQ)
        audio_np = resample_poly(audio_tensor.numpy(), ENCODEC_SFREQ // g, sr // g)
        audio_tensor = torch.from_numpy(audio_np).float()

    return audio_tensor


def encode_song(model: EncodecModel, audio: torch.Tensor) -> torch.Tensor:
    """Encode a full song through EnCodec's encoder.

    Args:
        audio: (T,) mono audio at 24kHz

    Returns:
        (n_windows, 128, 75) tensor of encoder embeddings per 1-second window
    """
    # EnCodec encoder expects (batch, channels, time)
    audio_input = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    with torch.no_grad():
        encoder_output = model.encoder(audio_input)  # (1, 128, T_frames)

    encoder_output = encoder_output.squeeze(0)  # (128, T_frames)

    # Number of complete 1-second windows (75 frames each)
    # Must match the EEG windowing: n_windows = n_eeg_samples // NMED_SFREQ
    n_frames = encoder_output.shape[1]
    n_windows = n_frames // FRAMES_PER_SECOND

    # Trim to complete windows and reshape
    usable_frames = n_windows * FRAMES_PER_SECOND
    encoder_output = encoder_output[:, :usable_frames]
    windows = encoder_output.reshape(128, n_windows, FRAMES_PER_SECOND)
    windows = windows.permute(1, 0, 2)  # (n_windows, 128, 75)

    return windows


def build():
    print("=== NMED-T EnCodec Embedding Builder ===")

    # Load EnCodec model
    print("\n[1/3] Loading EnCodec model...")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("  Using CUDA")

    # Pre-compute embeddings per song
    print("\n[2/3] Encoding songs...")
    song_embeds: dict[int, torch.Tensor] = {}  # song_id -> (n_windows, 128, 75)

    for song_id in tqdm(range(1, 11), desc="Encoding songs"):
        song = SONG_BY_ID[song_id]
        audio = load_audio_24khz(song_id)

        if torch.cuda.is_available():
            audio = audio.cuda()

        windows = encode_song(model, audio)
        song_embeds[song_id] = windows.cpu()
        print(f"  Song {song_id} ({song.title}): {windows.shape[0]} windows")

    # Build per-split embedding tensors aligned with EEG metadata ordering
    print("\n[3/3] Building aligned tensors...")

    for split in ["train", "val"]:
        meta_path = NMED_OUTPUT_DIR / f"nmed-{split}-metadata.parquet"
        meta = pl.read_parquet(meta_path)

        n_samples = len(meta)
        audio_embeds = torch.zeros(n_samples, 128, FRAMES_PER_SECOND)

        for i in range(n_samples):
            row = meta.row(i, named=True)
            song_id = row["song_id"]
            window_idx = row["window_idx"]
            audio_embeds[i] = song_embeds[song_id][window_idx]

        out_path = NMED_OUTPUT_DIR / f"nmed-{split}-encodec.safetensors"
        save_file({"audio_embeds": audio_embeds}, out_path)
        print(f"  Saved {split}: {audio_embeds.shape} to {out_path}")


if __name__ == "__main__":
    build()
