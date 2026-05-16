#!/usr/bin/env python3
"""
Frame Composite Tool
====================
Composites sampled frames from a video to enhance static elements (e.g. scoreboards).
Static regions survive averaging; moving objects blur away.

Usage:
    python frame_composite.py input.mp4 -n 30 -o output_dir/
    python frame_composite.py input.mp4 --methods mean median --resize 960
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ── Frame Extraction ──────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

def extract_frames(video_path: str, n_frames: int, resize_width: int | None = None) -> list[np.ndarray]:
    """Uniformly sample n_frames from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Error: cannot open '{video_path}'")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        sys.exit("Error: video has fewer than 2 frames")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    # deduplicate while preserving order
    indices = list(dict.fromkeys(indices))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if resize_width:
            h, w = frame.shape[:2]
            new_h = int(h * resize_width / w)
            frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    print(f"  Extracted {len(frames)} frames (requested {n_frames}, video has {total})")
    return frames


def collect_video_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
    sys.exit(f"Error: input path does not exist: '{input_path}'")


def process_video(video_path: Path, out_dir: Path, n_frames: int, methods_to_run: list[str],
                  resize_width: int | None = None, stability_threshold: float = 15.0,
                  sigma_clip_k: float = 2.0, sigma_clip_iter: int = 3,
                  bin_width: int = 20, save_reference: bool = True, output_prefix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"Sampling {n_frames} frames, output → {out_dir}/")
    frames = extract_frames(str(video_path), n_frames, resize_width)

    if len(frames) < 3:
        print("  [跳過] 需要至少 3 張 frames 才能產生有意義的 composite")
        return

    if save_reference:
        mid = len(frames) // 2
        single_path = out_dir / f"{output_prefix}00_single_frame.png"
        cv2.imwrite(str(single_path), frames[mid])
        print(f"  Saved reference single frame → {single_path}")

    for key in methods_to_run:
        label, func = ALL_METHODS[key]
        print(f"  Computing {label}...", end=" ", flush=True)
        if key == "stability_mask":
            result = func(frames, threshold=stability_threshold)
        elif key == "sigma_clip":
            result = func(frames, sigma=sigma_clip_k, iterations=sigma_clip_iter)
        elif key == "dominant_cluster":
            result = func(frames, bin_width=bin_width)
        else:
            result = func(frames)
        out_path = out_dir / f"{output_prefix}{key}.png"
        cv2.imwrite(str(out_path), result)
        print(f"→ {out_path}")

    if save_reference:
        print(f"  Done: {len(methods_to_run)} composites + 1 reference saved to {out_dir}/")
    else:
        print(f"  Done: {len(methods_to_run)} composites saved to {out_dir}/")


# ── Composite Methods ─────────────────────────────────────────────────────────

def composite_mean(frames: list[np.ndarray]) -> np.ndarray:
    """Pixel-wise mean. Static regions stay sharp; moving regions blur."""
    stack = np.stack(frames).astype(np.float64)
    return np.mean(stack, axis=0).astype(np.uint8)


def composite_median(frames: list[np.ndarray]) -> np.ndarray:
    """Pixel-wise median. More robust to occasional occlusions than mean."""
    stack = np.stack(frames)
    return np.median(stack, axis=0).astype(np.uint8)


def composite_max(frames: list[np.ndarray]) -> np.ndarray:
    """Pixel-wise max — keeps the brightest value seen at each pixel.
    Useful if the scoreboard is brighter than the background."""
    stack = np.stack(frames)
    return np.max(stack, axis=0).astype(np.uint8)


def composite_min(frames: list[np.ndarray]) -> np.ndarray:
    """Pixel-wise min — keeps the darkest value seen at each pixel.
    Useful if the scoreboard is darker than the background."""
    stack = np.stack(frames)
    return np.min(stack, axis=0).astype(np.uint8)


def composite_trimmed_mean(frames: list[np.ndarray], trim_pct: float = 0.1) -> np.ndarray:
    """Trimmed mean: discard the top/bottom trim_pct of values per pixel,
    then average the rest. Balances mean's smoothness with median's robustness."""
    stack = np.stack(frames).astype(np.float64)
    n = stack.shape[0]
    trim_count = max(1, int(n * trim_pct))
    sorted_stack = np.sort(stack, axis=0)
    trimmed = sorted_stack[trim_count : n - trim_count]
    return np.mean(trimmed, axis=0).astype(np.uint8)


def composite_std_map(frames: list[np.ndarray]) -> np.ndarray:
    """Standard deviation heatmap — NOT a composite, but a diagnostic.
    Dark = static (scoreboard), bright = high motion.
    Useful for visually confirming which regions are stable."""
    stack = np.stack(frames).astype(np.float64)
    std = np.std(stack, axis=0)
    # average across color channels for a single-channel map
    std_gray = np.mean(std, axis=2)
    # normalize to 0-255
    std_norm = ((std_gray - std_gray.min()) / (std_gray.max() - std_gray.min() + 1e-8) * 255).astype(np.uint8)
    return cv2.applyColorMap(std_norm, cv2.COLORMAP_INFERNO)


def composite_stability_mask(frames: list[np.ndarray], threshold: float = 15.0) -> np.ndarray:
    """Shows only the stable regions (std < threshold) from the median composite.
    Everything else is blacked out. Good for isolating the scoreboard."""
    stack = np.stack(frames).astype(np.float64)
    std = np.std(stack, axis=0)
    std_gray = np.mean(std, axis=2)
    mask = (std_gray < threshold).astype(np.uint8)

    median_img = composite_median(frames)
    result = median_img.copy()
    result[mask == 0] = 0
    return result


def composite_sigma_clip(frames: list[np.ndarray], sigma: float = 2.0, iterations: int = 3) -> np.ndarray:
    """Sigma-clipped mean (from astrophotography): iteratively compute mean & std,
    reject pixels beyond k·σ, recompute on survivors, then average what remains.
    Good at removing transient noise/occlusions when the 'true' signal is the majority.
    NOTE: if the overlay appears in <50% of frames, it may be rejected as outliers."""
    stack = np.stack(frames).astype(np.float64)  # (N, H, W, 3)
    N, H, W, C = stack.shape

    # Use grayscale for outlier decisions, apply mask to full color
    gray = np.mean(stack, axis=3)  # (N, H, W)
    mask = np.ones((N, H, W), dtype=bool)

    for _ in range(iterations):
        valid_count = np.maximum(mask.sum(axis=0), 1)  # (H, W)

        # Masked mean & std on grayscale
        gray_masked = gray * mask
        mu = gray_masked.sum(axis=0) / valid_count
        diff_sq = ((gray - mu[np.newaxis]) ** 2) * mask
        sd = np.sqrt(diff_sq.sum(axis=0) / valid_count)
        sd = np.maximum(sd, 1.0)  # prevent zero-std from killing everything

        # Reject outliers
        deviation = np.abs(gray - mu[np.newaxis])
        mask = mask & (deviation < sigma * sd[np.newaxis])

    # Average survivors (full color)
    mask_4d = mask[:, :, :, np.newaxis]
    masked_sum = (stack * mask_4d).sum(axis=0)
    count = np.maximum(mask.sum(axis=0), 1)[:, :, np.newaxis]
    return (masked_sum / count).astype(np.uint8)


def composite_dominant_cluster(frames: list[np.ndarray], bin_width: int = 20) -> np.ndarray:
    """Dominant-cluster mean: for each pixel, bin the N values by luminance,
    find the most populated bin, and average only those values.

    Designed for overlay scoreboards that may appear in ANY fraction of frames
    (even <50%). The overlay pixel values cluster tightly (same text color every
    frame), while background values scatter (players moving). The tightest
    cluster wins regardless of how many frames it spans."""
    stack = np.stack(frames).astype(np.float64)  # (N, H, W, 3)
    N, H, W, C = stack.shape

    # Bin on grayscale luminance
    gray = np.mean(stack, axis=3)                # (N, H, W)
    binned = (gray / bin_width).astype(np.int32)  # (N, H, W)

    # Count each bin at every pixel — only ~13 bins for bin_width=20, fully vectorized
    max_bin = int(255 / bin_width) + 2
    counts = np.zeros((max_bin, H, W), dtype=np.int32)
    for b in range(max_bin):
        counts[b] = (binned == b).sum(axis=0)

    # Dominant bin per pixel
    dominant = np.argmax(counts, axis=0)  # (H, W)

    # Keep only frames matching the dominant bin at each pixel
    mask = (binned == dominant[np.newaxis, :, :])  # (N, H, W)

    # Average matching frames (full color)
    mask_4d = mask[:, :, :, np.newaxis]
    masked_sum = (stack * mask_4d).sum(axis=0)
    count = np.maximum(mask.sum(axis=0), 1)[:, :, np.newaxis]
    return (masked_sum / count).astype(np.uint8)


ALL_METHODS = {
    "mean":              ("Mean Average", composite_mean),
    "median":            ("Median", composite_median),
    "trimmed_mean":      ("Trimmed Mean (10%)", composite_trimmed_mean),
    "max":               ("Pixel-wise Max", composite_max),
    "min":               ("Pixel-wise Min", composite_min),
    "sigma_clip":        ("Sigma-Clipped Mean", composite_sigma_clip),
    "dominant_cluster":  ("Dominant Cluster Mean", composite_dominant_cluster),
    "std_map":           ("Std Dev Heatmap (diagnostic)", composite_std_map),
    "stability_mask":    ("Stable Regions Only (median + mask)", composite_stability_mask),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Composite video frames to enhance static elements")
    parser.add_argument("input", help="Input video file or a folder containing videos")
    parser.add_argument("-n", "--n-frames", type=int, default=30,
                        help="Number of frames to sample (default: 30)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default <input_stem>_composites/)")
    parser.add_argument("--methods", nargs="+", choices=list(ALL_METHODS.keys()),
                        default=None, help="Methods to run (default: all)")
    parser.add_argument("--resize", type=int, default=None,
                        help="Resize frames to this width (preserves aspect ratio)")
    parser.add_argument("--stability-threshold", type=float, default=15.0,
                        help="Std threshold for stability_mask (default: 15.0)")
    parser.add_argument("--sigma-clip-k", type=float, default=2.0,
                        help="Sigma multiplier for sigma_clip (default: 2.0)")
    parser.add_argument("--sigma-clip-iter", type=int, default=3,
                        help="Iterations for sigma_clip (default: 3)")
    parser.add_argument("--bin-width", type=int, default=20,
                        help="Bin width for dominant_cluster, in pixel intensity 0-255 (default: 20)")
    args = parser.parse_args()

    input_path = Path(args.input)
    video_files = collect_video_files(input_path)
    if not video_files:
        sys.exit("Error: no video files found")

    is_batch_mode = input_path.is_dir()
    out_root = Path(args.output_dir) if args.output_dir else Path(f"{input_path.stem}_composites")
    out_root.mkdir(parents=True, exist_ok=True)

    methods_to_run = args.methods or list(ALL_METHODS.keys())

    if is_batch_mode:
        print(f"Input folder: {input_path}")
        print(f"Found {len(video_files)} video(s), output root → {out_root}/")
        print()
        for video_path in video_files:
            prefix = f"{video_path.stem}_"
            process_video(
                video_path=video_path,
                out_dir=out_root,
                n_frames=args.n_frames,
                methods_to_run=methods_to_run,
                resize_width=args.resize,
                stability_threshold=args.stability_threshold,
                sigma_clip_k=args.sigma_clip_k,
                sigma_clip_iter=args.sigma_clip_iter,
                bin_width=args.bin_width,
                save_reference=False,
                output_prefix=prefix,
            )
            print()
        print(f"Done! Batch composites saved under {out_root}/")
    else:
        process_video(
            video_path=input_path,
            out_dir=out_root,
            n_frames=args.n_frames,
            methods_to_run=methods_to_run,
            resize_width=args.resize,
            stability_threshold=args.stability_threshold,
            sigma_clip_k=args.sigma_clip_k,
            sigma_clip_iter=args.sigma_clip_iter,
            bin_width=args.bin_width,
            save_reference=True,
            output_prefix="",
        )
        print(f"\nDone! {len(methods_to_run)} composites + 1 reference saved to {out_root}/")


if __name__ == "__main__":
    main()
