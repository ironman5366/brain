"""
Pre-compute EnCodec encoder embeddings for HBN movie audio.

Produces per-split audio_embeds tensors aligned 1:1 with EEG samples,
matching the standard format used by all other datasets.

Usage:
    uv run python -m data.hbn.build_encodec
"""

from math import gcd

import polars as pl
import soundfile as sf
import torch
from safetensors.torch import save_file
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import EncodecModel

from data.hbn.movies import HBN_AUDIO_DIR, HBN_OUTPUT_DIR, MOVIES

ENCODEC_SFREQ = 24000
ENCODEC_HOP = 320  # 24000/320 = 75 frames/sec
FRAMES_PER_SECOND = ENCODEC_SFREQ // ENCODEC_HOP  # 75
WINDOW_SECONDS = 1.0


def load_audio_24khz(task_name: str) -> torch.Tensor:
    """Load a movie's extracted audio at 24kHz mono."""
    path = HBN_AUDIO_DIR / f"{task_name}.wav"
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio_tensor = torch.from_numpy(audio).float()

    if sr != ENCODEC_SFREQ:
        g = gcd(sr, ENCODEC_SFREQ)
        audio_np = resample_poly(audio_tensor.numpy(), ENCODEC_SFREQ // g, sr // g)
        audio_tensor = torch.from_numpy(audio_np).float()

    return audio_tensor


def encode_audio(model: EncodecModel, audio: torch.Tensor) -> torch.Tensor:
    """Encode audio through EnCodec's encoder.

    Args:
        audio: (T,) mono audio at 24kHz

    Returns:
        (n_windows, 128, 75) tensor of encoder embeddings per 1-second window
    """
    audio_input = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    with torch.no_grad():
        encoder_output = model.encoder(audio_input)  # (1, 128, T_frames)

    encoder_output = encoder_output.squeeze(0)  # (128, T_frames)

    n_frames = encoder_output.shape[1]
    n_windows = n_frames // FRAMES_PER_SECOND

    usable_frames = n_windows * FRAMES_PER_SECOND
    encoder_output = encoder_output[:, :usable_frames]
    windows = encoder_output.reshape(128, n_windows, FRAMES_PER_SECOND)
    windows = windows.permute(1, 0, 2)  # (n_windows, 128, 75)

    return windows


def build():
    print("=== HBN Movie EnCodec Embedding Builder ===")

    # Load EnCodec model
    print("\n[1/3] Loading EnCodec model...")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("  Using CUDA")

    # Encode each movie
    print("\n[2/3] Encoding movie audio tracks...")
    movie_embeds: dict[str, torch.Tensor] = {}  # task_name -> (n_windows, 128, 75)

    for movie in MOVIES:
        audio = load_audio_24khz(movie.task_name)

        if torch.cuda.is_available():
            audio = audio.cuda()

        windows = encode_audio(model, audio)
        movie_embeds[movie.task_name] = windows.cpu()
        print(f"  {movie.task_name}: {windows.shape[0]} windows ({movie.stimulus_duration:.1f}s stimulus)")

    # Build per-split embedding tensors aligned with EEG metadata ordering
    print("\n[3/3] Building aligned tensors...")

    for split in ["train", "val"]:
        meta_path = HBN_OUTPUT_DIR / f"hbn-{split}-metadata.parquet"
        if not meta_path.exists():
            print(f"  Skipping {split}: metadata not found at {meta_path}")
            continue

        meta = pl.read_parquet(meta_path)
        n_samples = len(meta)
        audio_embeds = torch.zeros(n_samples, 128, FRAMES_PER_SECOND)

        skipped = 0
        for i in range(n_samples):
            row = meta.row(i, named=True)
            task_name = row["task_name"]
            window_idx = row["window_idx"]

            if task_name not in movie_embeds:
                skipped += 1
                continue

            embeds = movie_embeds[task_name]
            if window_idx < embeds.shape[0]:
                audio_embeds[i] = embeds[window_idx]
            else:
                skipped += 1

        if skipped > 0:
            print(f"  WARNING: {skipped} samples had missing audio embeddings")

        out_path = HBN_OUTPUT_DIR / f"hbn-{split}-encodec.safetensors"
        save_file({"audio_embeds": audio_embeds}, out_path)
        print(f"  Saved {split}: {audio_embeds.shape} to {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    build()
