#!/usr/bin/env python3
"""
Download DAVIS 2016 and DAVIS 2017 datasets.

Downloads all sequences from:
- DAVIS 2017 trainval (includes all DAVIS 2016 sequences)
- DAVIS 2017 test-dev
- DAVIS 2017 test-challenge

Usage:
    python scripts/download_davis.py /path/to/download/location

The script will create the following structure:
    /path/to/download/location/
        JPEGImages/
            480p/
                bear/
                blackswan/
                ...
"""

import argparse
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path


DAVIS_URLS = {
    "trainval": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
    "test-dev": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip",
    "test-challenge": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip",
}


def download_file(url: str, dest_path: Path, desc: str) -> bool:
    """download a file with progress indicator and time estimates."""
    print(f"downloading {desc}...")
    print(f"  url: {url}")
    print(f"  destination: {dest_path}")

    start_time = time.time()

    try:
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)

                elapsed = time.time() - start_time
                if downloaded > 0 and elapsed > 0:
                    speed_mbps = mb_downloaded / elapsed
                    remaining_mb = mb_total - mb_downloaded
                    eta_seconds = remaining_mb / speed_mbps if speed_mbps > 0 else 0
                    eta_str = f"{int(eta_seconds)}s" if eta_seconds < 60 else f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    print(f"\r  progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB) - {speed_mbps:.1f} MB/s - ETA: {eta_str}   ", end="", flush=True)
                else:
                    print(f"\r  progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        elapsed = time.time() - start_time
        print(f"\n  completed in {elapsed:.1f}s")
        return True
    except urllib.error.HTTPError as e:
        print(f"\n  error: HTTP {e.code} - {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n  error: {e.reason}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """extract a zip file."""
    print(f"  extracting {zip_path.name}...", end="", flush=True)
    start = time.time()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f" done ({time.time() - start:.1f}s)")


def merge_sequences(src_dir: Path, dest_dir: Path) -> dict:
    """
    merge DAVIS sequences from src to dest, avoiding duplicates.

    returns dict with counts of new, skipped (duplicate), and total sequences.
    """
    jpeg_src = src_dir / "DAVIS" / "JPEGImages" / "480p"
    if not jpeg_src.exists():
        # try alternate structure
        jpeg_src = src_dir / "JPEGImages" / "480p"

    if not jpeg_src.exists():
        print(f"  warning: no JPEGImages/480p found in {src_dir}")
        return {"new": 0, "skipped": 0, "total": 0}

    jpeg_dest = dest_dir / "JPEGImages" / "480p"
    jpeg_dest.mkdir(parents=True, exist_ok=True)

    new_count = 0
    skipped_count = 0

    for seq_dir in sorted(jpeg_src.iterdir()):
        if seq_dir.is_dir():
            dest_seq = jpeg_dest / seq_dir.name
            if dest_seq.exists():
                skipped_count += 1
            else:
                shutil.copytree(seq_dir, dest_seq)
                new_count += 1

    return {"new": new_count, "skipped": skipped_count, "total": new_count + skipped_count}


def main():
    parser = argparse.ArgumentParser(
        description="Download DAVIS 2016 and DAVIS 2017 datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "download_location",
        type=str,
        help="Directory to download and extract DAVIS data to"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading if zip files already exist"
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep downloaded zip files after extraction"
    )

    args = parser.parse_args()

    dest_dir = Path(args.download_location).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("DAVIS dataset download")
    print("=" * 50)
    print(f"destination: {dest_dir}")
    print()

    # track total sequences
    total_sequences = set()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for name, url in DAVIS_URLS.items():
            print(f"\n[{name}]")

            zip_path = dest_dir / f"DAVIS-2017-{name}-480p.zip"

            # download
            if zip_path.exists() and args.skip_existing:
                print(f"  skipping download (zip exists): {zip_path}")
            else:
                success = download_file(url, zip_path, f"DAVIS 2017 {name}")
                if not success:
                    print(f"  skipping {name} due to download error")
                    continue

            # extract to temp dir
            extract_dir = tmpdir / name
            extract_dir.mkdir(exist_ok=True)
            extract_zip(zip_path, extract_dir)

            # merge sequences
            stats = merge_sequences(extract_dir, dest_dir)
            print(f"  sequences: {stats['new']} new, {stats['skipped']} already existed")

            # track sequences
            jpeg_dir = dest_dir / "JPEGImages" / "480p"
            if jpeg_dir.exists():
                for seq in jpeg_dir.iterdir():
                    if seq.is_dir():
                        total_sequences.add(seq.name)

            # cleanup zip if requested
            if not args.keep_zips and zip_path.exists():
                zip_path.unlink()
                print(f"  removed: {zip_path.name}")

    # final summary
    print()
    print("=" * 50)
    print("download complete!")
    print(f"total unique sequences: {len(total_sequences)}")
    print(f"location: {dest_dir / 'JPEGImages' / '480p'}")

    # count frames
    jpeg_dir = dest_dir / "JPEGImages" / "480p"
    if jpeg_dir.exists():
        total_frames = 0
        for seq in jpeg_dir.iterdir():
            if seq.is_dir():
                total_frames += len(list(seq.glob("*.jpg")))
        print(f"total frames: {total_frames}")

    print()
    print("to use with NeuralGraph, set in your config yaml:")
    print("  simulation:")
    print(f"    datavis_root: \"{dest_dir}\"")
    print()
    print("or set environment variable:")
    print(f"  export DATAVIS_ROOT=\"{dest_dir}\"")


if __name__ == "__main__":
    main()
