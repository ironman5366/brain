"""
Pre-compute EnCodec encoder embeddings for all DS002721 audio stimuli.

Produces audio_embeds tensors aligned with the EEG sample ordering
from the processed dataset.

Usage:
    uv run python -m data.musicemo.build_encodec
"""

from pathlib import Path

import polars as pl
import soundfile as sf
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import EncodecModel

from data.musicemo.songs import MUSICEMO_OUTPUT_DIR, MUSICEMO_STIMULI_DIR

ENCODEC_SFREQ = 24000
ENCODEC_HOP = 320  # 24000/320 = 75 frames/sec
FRAMES_PER_SECOND = ENCODEC_SFREQ // ENCODEC_HOP  # 75
WINDOW_SECONDS = 1.0


def load_audio_24khz(audio_path: Path) -> torch.Tensor:
    """Load audio and resample to 24kHz mono."""
    audio, sr = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio_tensor = torch.from_numpy(audio).float()

    if sr != ENCODEC_SFREQ:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(sr, ENCODEC_SFREQ)
        audio_np = resample_poly(audio_tensor.numpy(), ENCODEC_SFREQ // g, sr // g)
        audio_tensor = torch.from_numpy(audio_np).float()

    return audio_tensor


def encode_audio(model: EncodecModel, audio: torch.Tensor) -> torch.Tensor:
    """Encode audio through EnCodec's encoder.

    Returns:
        (n_windows, 128, 75) tensor of encoder embeddings per 1-second window
    """
    audio_input = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    with torch.no_grad():
        encoder_output = model.encoder(audio_input)  # (1, 128, T_frames)

    encoder_output = encoder_output.squeeze(0)  # (128, T_frames)

    n_frames = encoder_output.shape[1]
    n_windows = n_frames // FRAMES_PER_SECOND

    if n_windows == 0:
        return torch.zeros(0, 128, FRAMES_PER_SECOND)

    usable_frames = n_windows * FRAMES_PER_SECOND
    encoder_output = encoder_output[:, :usable_frames]
    windows = encoder_output.reshape(128, n_windows, FRAMES_PER_SECOND)
    windows = windows.permute(1, 0, 2)  # (n_windows, 128, 75)

    return windows


def build():
    print("=== DS002721 EnCodec Embedding Builder ===")

    # Load EnCodec model
    print("\n[1/3] Loading EnCodec model...")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("  Using CUDA")

    # Pre-compute embeddings per song
    print("\n[2/3] Encoding song stimuli...")
    song_embeds: dict[str, torch.Tensor] = {}  # filename -> (n_windows, 128, 75)

    mp3_files = sorted(MUSICEMO_STIMULI_DIR.glob("*.mp3"))
    print(f"  Found {len(mp3_files)} MP3 files")

    for mp3_path in tqdm(mp3_files, desc="Encoding"):
        audio = load_audio_24khz(mp3_path)

        if torch.cuda.is_available():
            audio = audio.cuda()

        windows = encode_audio(model, audio)
        song_embeds[mp3_path.name] = windows.cpu()

    # Build per-split embedding tensors aligned with EEG metadata ordering
    print("\n[3/3] Building aligned tensors...")

    for split in ["train", "val"]:
        meta_path = MUSICEMO_OUTPUT_DIR / f"musicemo-{split}-metadata.parquet"
        if not meta_path.exists():
            print(f"  Skipping {split}: metadata not found at {meta_path}")
            continue

        meta = pl.read_parquet(meta_path)
        n_samples = len(meta)
        audio_embeds = torch.zeros(n_samples, 128, FRAMES_PER_SECOND)

        skipped = 0
        for i in range(n_samples):
            row = meta.row(i, named=True)
            mp3_filename = row["mp3_filename"]
            window_idx = row["window_idx"]

            if mp3_filename not in song_embeds:
                skipped += 1
                continue

            embeds = song_embeds[mp3_filename]
            if window_idx < embeds.shape[0]:
                audio_embeds[i] = embeds[window_idx]
            else:
                skipped += 1

        if skipped > 0:
            print(f"  WARNING: {skipped} samples had missing audio embeddings")

        out_path = MUSICEMO_OUTPUT_DIR / f"musicemo-{split}-encodec.safetensors"
        save_file({"audio_embeds": audio_embeds}, out_path)
        print(f"  Saved {split}: {audio_embeds.shape} to {out_path}")


if __name__ == "__main__":
    build()
