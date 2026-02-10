"""
Pre-compute EnCodec encoder embeddings for EAV audio.

Reads the pre-extracted audio files from eav-audio/ (produced by build.py)
and encodes them through EnCodec's encoder. Produces audio_embeds tensors
aligned with the EEG sample ordering from the processed dataset.

Usage:
    uv run python -m data.eav.build_encodec                  # sparse (eav)
    uv run python -m data.eav.build_encodec --prefix eav-32ch # dense 32ch
"""

from math import gcd

import polars as pl
import soundfile as sf
import torch
from safetensors.torch import save_file
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import EncodecModel

from data.eav.songs import EAV_AUDIO_DIR, EAV_OUTPUT_DIR

ENCODEC_SFREQ = 24000
ENCODEC_HOP = 320  # downsampling factor: 24000/320 = 75 frames/sec
FRAMES_PER_SECOND = ENCODEC_SFREQ // ENCODEC_HOP  # 75
WINDOW_SECONDS = 1.0


def load_audio_24khz(audio_filename: str) -> torch.Tensor:
    """Load an audio file from the pre-extracted dir and resample to 24kHz mono."""
    path = EAV_AUDIO_DIR / audio_filename
    audio, sr = sf.read(str(path))

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

    Returns:
        (n_windows, 128, 75) tensor of encoder embeddings per 1-second window
    """
    audio_input = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    with torch.no_grad():
        if next(model.parameters()).is_cuda:
            audio_input = audio_input.cuda()
        encoder_output = model.encoder(audio_input)  # (1, 128, T_frames)

    encoder_output = encoder_output.squeeze(0).cpu()  # (128, T_frames)

    n_frames = encoder_output.shape[1]
    n_windows = n_frames // FRAMES_PER_SECOND

    usable_frames = n_windows * FRAMES_PER_SECOND
    encoder_output = encoder_output[:, :usable_frames]
    windows = encoder_output.reshape(128, n_windows, FRAMES_PER_SECOND)
    windows = windows.permute(1, 0, 2)  # (n_windows, 128, 75)

    return windows


def build(prefix: str = "eav"):
    print(f"=== EAV EnCodec Embedding Builder (prefix={prefix}) ===")

    print("\n[1/2] Loading EnCodec model...")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("  Using CUDA")

    print("\n[2/2] Encoding audio per trial...")

    for split in ["train", "val"]:
        meta_path = EAV_OUTPUT_DIR / f"{prefix}-{split}-metadata.parquet"
        if not meta_path.exists():
            print(f"  Skipping {split}: metadata not found at {meta_path}")
            continue

        meta = pl.read_parquet(meta_path)
        n_samples = len(meta)
        audio_embeds = torch.zeros(n_samples, 128, FRAMES_PER_SECOND)

        # Cache: audio_filename -> (n_windows, 128, 75)
        trial_cache: dict[str, torch.Tensor | None] = {}

        n_found = 0
        n_missing = 0

        for i in tqdm(range(n_samples), desc=f"Encoding {split}"):
            row = meta.row(i, named=True)
            audio_file = row["audio_filename"]
            window_idx = row["window_idx"]

            if audio_file not in trial_cache:
                audio_path = EAV_AUDIO_DIR / audio_file
                if audio_path.exists():
                    audio = load_audio_24khz(audio_file)
                    trial_cache[audio_file] = encode_audio(model, audio)
                else:
                    trial_cache[audio_file] = None

            embeds = trial_cache[audio_file]
            if embeds is not None and window_idx < embeds.shape[0]:
                audio_embeds[i] = embeds[window_idx]
                n_found += 1
            else:
                n_missing += 1

        out_path = EAV_OUTPUT_DIR / f"{prefix}-{split}-encodec.safetensors"
        save_file({"audio_embeds": audio_embeds}, out_path)
        print(f"  Saved {split}: {audio_embeds.shape} to {out_path}")
        print(f"    Audio found: {n_found:,}, missing/short: {n_missing:,}")

    print("\nDone!")


if __name__ == "__main__":
    import sys
    pfx = "eav"
    if "--prefix" in sys.argv:
        idx = sys.argv.index("--prefix")
        if idx + 1 < len(sys.argv):
            pfx = sys.argv[idx + 1]
    build(prefix=pfx)
