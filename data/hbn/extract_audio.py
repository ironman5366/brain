"""
Extract and trim audio from HBN movie video files.

Produces 24kHz mono WAV files for each movie stimulus, trimmed to match
the stimulus presentation timing used in the HBN EEG protocol.

Usage:
    uv run python -m data.hbn.extract_audio
"""

import subprocess

from data.hbn.movies import HBN_AUDIO_DIR, HBN_VIDEO_DIR, MOVIES


def extract_audio():
    HBN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    for movie in MOVIES:
        input_path = HBN_VIDEO_DIR / movie.video_filename
        output_path = HBN_AUDIO_DIR / f"{movie.task_name}.wav"

        if output_path.exists():
            print(f"  Skipping {movie.task_name} (already exists)")
            continue

        if not input_path.exists():
            print(f"  WARNING: Video not found: {input_path}")
            continue

        cmd = [
            "ffmpeg",
            "-v", "error",
        ]

        # Trim start
        if movie.trim_start > 0:
            cmd += ["-ss", str(movie.trim_start)]

        cmd += ["-i", str(input_path)]

        # Trim end
        if movie.trim_end is not None:
            if movie.trim_start > 0:
                # -to is relative to -ss when -ss is before -i
                cmd += ["-to", str(movie.trim_end - movie.trim_start)]
            else:
                cmd += ["-to", str(movie.trim_end)]

        # Output: 24kHz mono WAV (no video)
        cmd += [
            "-vn",
            "-ac", "1",
            "-ar", "24000",
            str(output_path),
        ]

        print(f"  Extracting {movie.task_name}: {input_path.name}")
        print(f"    trim: {movie.trim_start}s - {movie.trim_end or 'end'}")
        subprocess.run(cmd, check=True)
        print(f"    -> {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    extract_audio()
