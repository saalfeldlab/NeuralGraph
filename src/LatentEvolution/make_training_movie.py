"""
Convert TensorBoard logged figures into a movie.

Usage:
    python make_training_movie.py <run_dir> <plot_tag>

Example:
    python make_training_movie.py runs/my_run Val/multi_start_2000step_latent_rollout_mses_by_time
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing import event_accumulator


def sanitize_filename(tag: str) -> str:
    """Convert a tensorboard tag to a valid filename."""
    return re.sub(r'[^\w\-.]', '_', tag)


def add_epoch_overlay(image: np.ndarray, epoch: int) -> np.ndarray:
    """Add epoch number overlay to image."""
    # convert to PIL for text rendering
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # try to use a monospace font, fall back to default
    font_size = 32
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    text = f"epoch {epoch}"

    # get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # position in top-right corner with padding
    padding = 10
    x = image.shape[1] - text_width - padding - 10
    y = padding

    # draw background rectangle
    draw.rectangle(
        [x - 5, y - 2, x + text_width + 5, y + text_height + 5],
        fill=(0, 0, 0, 200)
    )

    # draw text
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return np.array(pil_image)


def load_images_from_tensorboard(run_dir: Path, tag: str) -> list[tuple[int, np.ndarray]]:
    """Load all images for a given tag from tensorboard event files."""
    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.IMAGES: 0}  # load all images
    )
    ea.Reload()

    available_tags = ea.Tags().get('images', [])
    if tag not in available_tags:
        print(f"tag '{tag}' not found. available image tags:")
        for t in sorted(available_tags):
            print(f"  {t}")
        raise ValueError(f"tag '{tag}' not found")

    images = []
    for event in ea.Images(tag):
        # decode image from png bytes
        img_bytes = event.encoded_image_string
        pil_image = Image.open(io.BytesIO(img_bytes))
        # convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        np_image = np.array(pil_image)
        images.append((event.step, np_image))

    # sort by step (epoch)
    images.sort(key=lambda x: x[0])
    return images


def create_movie(
    images: list[tuple[int, np.ndarray]],
    output_path: Path,
    fps: int = 2,
) -> None:
    """Create a compressed movie from images with epoch overlay."""
    if not images:
        raise ValueError("no images to create movie from")

    # create temp directory for frames
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # save frames as png files
        for i, (epoch, image) in enumerate(images):
            frame = add_epoch_overlay(image, epoch)
            frame_path = tmpdir / f"frame_{i:06d}.png"
            Image.fromarray(frame).save(frame_path)

        # use ffmpeg to create h264 video (QuickTime compatible)
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",  # required for QuickTime
            "-crf", "23",  # quality (lower = better, 18-28 is reasonable)
            "-preset", "medium",
            str(output_path),
        ]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    print(f"wrote {len(images)} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TensorBoard logged figures into a movie.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python make_training_movie.py runs/my_run Val/multi_start_2000step_latent_rollout_mses_by_time
    python make_training_movie.py runs/my_run CrossVal/fly_N9_62_0/best_2000step_rollout_latent_mse_var_scatter
        """
    )
    parser.add_argument("run_dir", type=Path, help="path to the run directory")
    parser.add_argument("tag", type=str, help="tensorboard image tag (e.g., Val/plot_name)")
    parser.add_argument("--fps", type=int, default=2, help="frames per second (default: 2)")
    parser.add_argument("--list-tags", action="store_true", help="list available image tags and exit")
    parser.add_argument("--cwd", action="store_true", help="output movie to current working directory instead of run directory")

    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {args.run_dir}")

    # list tags mode
    if args.list_tags:
        ea = event_accumulator.EventAccumulator(
            str(args.run_dir),
            size_guidance={event_accumulator.IMAGES: 0}
        )
        ea.Reload()
        print("available image tags:")
        for t in sorted(ea.Tags().get('images', [])):
            print(f"  {t}")
        return

    # load images
    print(f"loading images from {args.run_dir} with tag '{args.tag}'...")
    images = load_images_from_tensorboard(args.run_dir, args.tag)
    print(f"loaded {len(images)} images")

    # create output path
    output_filename = sanitize_filename(args.tag) + ".mp4"
    output_dir = Path.cwd() if args.cwd else args.run_dir
    output_path = output_dir / output_filename

    # create movie
    print(f"creating movie at {output_path}...")
    create_movie(images, output_path, fps=args.fps)

    # print file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"output size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
