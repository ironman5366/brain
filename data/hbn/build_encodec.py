"""
Pre-compute EnCodec encoder embeddings for HBN movie audio.

Since all subjects watch the same 4 movies, this only encodes 4 audio tracks
and saves per-movie embedding tensors. The HBNAudioEmbedDataset class handles
lookup at runtime, avoiding ~60 GB of duplicated data.

Usage:
    uv run python -m data.hbn.build_encodec
"""

from math import gcd

import soundfile as sf
import torch
from safetensors.torch import save_file
from scipy.signal import resample_poly
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
    print("\n[1/2] Loading EnCodec model...")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("  Using CUDA")

    # Encode each movie
    print("\n[2/2] Encoding movie audio tracks...")
    movie_embeds = {}

    for movie in MOVIES:
        audio = load_audio_24khz(movie.task_name)

        if torch.cuda.is_available():
            audio = audio.cuda()

        windows = encode_audio(model, audio)
        movie_embeds[movie.task_name] = windows.cpu().contiguous()
        print(f"  {movie.task_name}: {windows.shape[0]} windows ({movie.stimulus_duration:.1f}s stimulus)")

    # Save per-movie embeddings in a single file
    out_path = HBN_OUTPUT_DIR / "hbn-movie-encodec.safetensors"
    HBN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(movie_embeds, out_path)
    print(f"\n  Saved to {out_path}")

    total_windows = sum(t.shape[0] for t in movie_embeds.values())
    print(f"  Total: {total_windows} unique audio windows across 4 movies")

    print("\nDone!")


if __name__ == "__main__":
    build()
