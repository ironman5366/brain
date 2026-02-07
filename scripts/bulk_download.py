#!/usr/bin/env python3
"""Bulk download files in parallel with progress bars."""

import argparse
import asyncio
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiohttp
from tqdm.asyncio import tqdm


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    position: int,
) -> tuple[str, bool, str]:
    """Download a single file with progress bar.

    Returns (filename, success, error_message).
    """
    async with semaphore:
        # Extract filename from URL
        parsed = urlparse(url)
        filename = unquote(Path(parsed.path).name)
        if not filename:
            filename = f"download_{position}"

        output_path = output_dir / filename

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=filename[:30],
                    position=position,
                    leave=True,
                ) as pbar:
                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            pbar.update(len(chunk))

                return filename, True, ""
        except aiohttp.ClientError as e:
            return filename, False, str(e)
        except Exception as e:
            return filename, False, str(e)


async def download_all(
    urls: list[str],
    output_dir: Path,
    max_concurrent: int = 5,
) -> list[tuple[str, bool, str]]:
    """Download all URLs in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)

    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=None, connect=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            download_file(session, url, output_dir, semaphore, i)
            for i, url in enumerate(urls)
        ]
        results = await asyncio.gather(*tasks)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download multiple files in parallel with progress bars."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input file with URLs (one per line). Use - for stdin.",
        default="-",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory (default: current directory)",
        default=".",
    )
    parser.add_argument(
        "-c",
        "--concurrent",
        type=int,
        help="Maximum concurrent downloads (default: 5)",
        default=5,
    )

    args = parser.parse_args()

    # Read URLs
    if args.input == "-":
        urls = [line.strip() for line in sys.stdin if line.strip()]
    else:
        with open(args.input) as f:
            urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print("No URLs provided.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(urls)} files to {output_dir.absolute()}")
    print()

    # Run downloads
    results = asyncio.run(download_all(urls, output_dir, args.concurrent))

    # Print summary
    print("\n" * len(urls))  # Clear space after progress bars
    print("=" * 50)
    print("Summary:")

    successes = [r for r in results if r[1]]
    failures = [r for r in results if not r[1]]

    print(f"  Successful: {len(successes)}")
    print(f"  Failed: {len(failures)}")

    if failures:
        print("\nFailed downloads:")
        for filename, _, error in failures:
            print(f"  - {filename}: {error}")


2
if __name__ == "__main__":
    main()
