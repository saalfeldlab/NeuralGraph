#!/usr/bin/env python3
"""
Download YouTube-VOS dataset from Google Drive.

Downloads the YouTube-VOS 2019 dataset which contains:
- Training: 3471 videos, 65 categories, 6459 object instances
- Validation: 507 videos, 65+26 categories, 1063 object instances

Usage:
    python scripts/download_youtube_vos.py /path/to/download/location

The script will create the following structure:
    /path/to/download/location/
        JPEGImages/
            480p/
                <video_id>/
                    00000.jpg
                    00005.jpg
                    ...

Requirements:
    pip install gdown
"""

import argparse
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path


# Google Drive folder containing YouTube-VOS dataset
# https://drive.google.com/drive/folders/1XwjQ-eysmOb7JdmJAwfVOBZX-aMbHccC
GDRIVE_FOLDER_ID = "1XwjQ-eysmOb7JdmJAwfVOBZX-aMbHccC"

# Known file IDs from the YouTube-VOS Google Drive (may need updating)
# These are typical files found in YouTube-VOS distributions
KNOWN_FILES = {
    # Add specific file IDs here if known, e.g.:
    # "train.zip": "1abc123...",
    # "valid.zip": "1def456...",
}


def check_gdown():
    """check if gdown is installed."""
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def install_gdown():
    """install gdown if not present."""
    print("installing gdown...")
    subprocess.run(["pip", "install", "gdown"], check=True)


def download_folder(folder_id: str, dest_dir: Path) -> bool:
    """download a google drive folder using gdown."""
    import gdown

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"downloading from: {url}")
    print(f"destination: {dest_dir}")
    print()
    print("note: this may take a while for large datasets (~30GB+)")
    print("gdown will show progress for each file...")
    print()

    start_time = time.time()

    try:
        gdown.download_folder(url, output=str(dest_dir), quiet=False, remaining_ok=True)
        elapsed = time.time() - start_time
        print(f"\ndownload completed in {elapsed/60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\nerror downloading folder: {e}")
        print("\nif you get rate limited, try:")
        print("  1. wait a few hours and retry")
        print("  2. manually download from the google drive link")
        print("  3. use a different network/vpn")
        return False


def download_file(file_id: str, dest_path: Path, desc: str) -> bool:
    """download a single file from google drive."""
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"downloading {desc}...")

    start_time = time.time()
    try:
        gdown.download(url, str(dest_path), quiet=False)
        elapsed = time.time() - start_time
        print(f"  completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"  error: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """extract tar or zip archive."""
    print(f"  extracting {archive_path.name}...", end="", flush=True)
    start = time.time()

    if archive_path.suffix == ".zip" or archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_to)
    elif archive_path.suffix in [".tar", ".gz", ".tgz"] or ".tar" in archive_path.name:
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(extract_to)
    else:
        print(f" unknown format: {archive_path.suffix}")
        return

    print(f" done ({time.time() - start:.1f}s)")


def reorganize_to_davis_format(src_dir: Path, dest_dir: Path) -> dict:
    """
    reorganize YouTube-VOS to DAVIS-like format.

    YouTube-VOS structure:
        train/JPEGImages/<video_id>/*.jpg
        valid/JPEGImages/<video_id>/*.jpg

    Target structure (DAVIS-like):
        JPEGImages/480p/<video_id>/*.jpg
    """
    jpeg_dest = dest_dir / "JPEGImages" / "480p"
    jpeg_dest.mkdir(parents=True, exist_ok=True)

    new_count = 0
    skipped_count = 0

    # look for JPEGImages in various locations
    search_paths = [
        src_dir / "train" / "JPEGImages",
        src_dir / "valid" / "JPEGImages",
        src_dir / "test" / "JPEGImages",
        src_dir / "train_all_frames" / "JPEGImages",
        src_dir / "valid_all_frames" / "JPEGImages",
        src_dir / "JPEGImages",
    ]

    for jpeg_src in search_paths:
        if not jpeg_src.exists():
            continue

        print(f"  processing {jpeg_src}...")

        for seq_dir in sorted(jpeg_src.iterdir()):
            if seq_dir.is_dir():
                dest_seq = jpeg_dest / seq_dir.name
                if dest_seq.exists():
                    skipped_count += 1
                else:
                    shutil.copytree(seq_dir, dest_seq)
                    new_count += 1

    return {"new": new_count, "skipped": skipped_count}


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube-VOS dataset from Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "download_location",
        type=str,
        help="Directory to download and extract YouTube-VOS data to"
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archive files after extraction"
    )
    parser.add_argument(
        "--skip-reorganize",
        action="store_true",
        help="Skip reorganizing to DAVIS-like format"
    )

    args = parser.parse_args()

    # check/install gdown
    if not check_gdown():
        install_gdown()

    dest_dir = Path(args.download_location).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("YouTube-VOS dataset download")
    print("=" * 50)
    print(f"destination: {dest_dir}")
    print(f"google drive folder: {GDRIVE_FOLDER_ID}")
    print()

    # download the folder
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        download_dir = tmpdir / "youtube_vos_raw"
        download_dir.mkdir()

        success = download_folder(GDRIVE_FOLDER_ID, download_dir)

        if not success:
            print("\ndownload failed. you can try manual download:")
            print(f"  1. go to: https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}")
            print("  2. download all files manually")
            print(f"  3. extract to: {dest_dir}")
            return 1

        # find and extract any archives
        print("\nlooking for archives to extract...")
        for archive in download_dir.rglob("*.tar"):
            extract_archive(archive, download_dir)
        for archive in download_dir.rglob("*.tar.gz"):
            extract_archive(archive, download_dir)
        for archive in download_dir.rglob("*.tgz"):
            extract_archive(archive, download_dir)
        for archive in download_dir.rglob("*.zip"):
            extract_archive(archive, download_dir)

        # reorganize to DAVIS-like format
        if not args.skip_reorganize:
            print("\nreorganizing to DAVIS-like format...")
            stats = reorganize_to_davis_format(download_dir, dest_dir)
            print(f"  sequences: {stats['new']} new, {stats['skipped']} already existed")
        else:
            # just copy everything
            print("\ncopying files...")
            for item in download_dir.iterdir():
                dest_item = dest_dir / item.name
                if item.is_dir():
                    if dest_item.exists():
                        shutil.rmtree(dest_item)
                    shutil.copytree(item, dest_item)
                else:
                    shutil.copy2(item, dest_item)

    # final summary
    print()
    print("=" * 50)
    print("download complete!")

    jpeg_dir = dest_dir / "JPEGImages" / "480p"
    if jpeg_dir.exists():
        sequences = [d for d in jpeg_dir.iterdir() if d.is_dir()]
        total_frames = sum(len(list(s.glob("*.jpg"))) for s in sequences)
        print(f"total sequences: {len(sequences)}")
        print(f"total frames: {total_frames}")
        print(f"location: {jpeg_dir}")

    print()
    print("to use with NeuralGraph, set in your config yaml:")
    print("  simulation:")
    print(f"    datavis_root: \"{dest_dir}\"")

    return 0


if __name__ == "__main__":
    exit(main())
