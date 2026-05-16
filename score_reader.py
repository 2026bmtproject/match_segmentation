#!/usr/bin/env python3
"""
Scoreboard Reader via Gemini Flash 2.5
=======================================
Reads badminton scoreboards from screenshots or video composites using Google Gemini API.

Usage:
    # Set API key (PowerShell) (https://aistudio.google.com/app/api-keys)
    $env:GEMINI_API_KEY = "your-key-here"

    # Single video
    python score_reader.py match.mp4

    # Batch folder of videos
    python score_reader.py clips/ -o results.csv

    # Single image still works
    python score_reader.py screenshot.png
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from frame_composite import (
    composite_dominant_cluster,
    composite_mean,
    composite_max,
    composite_median,
    composite_sigma_clip,
    extract_frames,
)

# ── Config ────────────────────────────────────────────────────────────────────

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

FALLBACK_METHODS = [
    ("dominant_cluster", composite_dominant_cluster),
    ("sigma_clip", composite_sigma_clip),
    ("median", composite_median),
    ("mean", composite_mean),
    ("max", composite_max),
]

CONFIDENCE_SCORE = {"high": 3, "medium": 2, "low": 1}

SCORE_PROMPT_SINGLE = """\
這是一場羽球比賽的截圖，請找出畫面中記分板上的比分。

羽球記分板格式說明：
- 記分板會顯示兩位選手的分數，每位選手後面會有 1~3 個數字
- 每個數字代表一局的分數，從左到右依序是第一局、第二局、第三局
- 最右邊的分數是「目前正在進行的那一局」的比分

請只回傳以下 JSON 格式，不要任何其他文字或 markdown：
{
  "score_a": <上方隊伍的整數分數>,
  "score_b": <下方隊伍的整數分數>,
  "confidence": "<high | medium | low>"
}

若完全看不到記分板，score_a 和 score_b 請填 null，confidence 填 "low"。\
"""

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_and_resize(path: str, max_width: int | None = None) -> bytes:
    """Load image, optionally resize, return JPEG bytes."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    if max_width and img.shape[1] > max_width:
        h, w = img.shape[:2]
        new_h = int(h * max_width / w)
        img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    return buf.tobytes()


def image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Encode a BGR image to JPEG bytes for Gemini."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Failed to encode composite image")
    return buf.tobytes()


def collect_inputs(input_path: str) -> list[Path]:
    """Return sorted list of files from a file or directory."""
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTS or p.suffix.lower() in SUPPORTED_VIDEO_EXTS:
            return [p]
        else:
            sys.exit(f"Error: unsupported file type '{p.suffix}'")
    elif p.is_dir():
        videos = sorted(
            f for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS
        )
        if videos:
            return videos

        images = sorted(
            f for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
        )
        if images:
            return images

        sys.exit(f"Error: no videos or images found in '{p}'")
    else:
        sys.exit(f"Error: '{input_path}' is not a file or directory")


def normalize_confidence(confidence: str | None) -> str:
    """Normalize model confidence into high/medium/low."""
    value = str(confidence or "low").strip().lower()
    if value not in CONFIDENCE_SCORE:
        return "low"
    return value


def candidate_rank(result: dict) -> tuple[int, int]:
    """Compare candidates by confidence first, then whether scores exist."""
    confidence = normalize_confidence(result.get("confidence"))
    score_present = int(result.get("score_a") is not None and result.get("score_b") is not None)
    return (CONFIDENCE_SCORE[confidence], score_present)


def format_attempts(attempts: list[dict]) -> str:
    """Compact attempt summary for CSV output."""
    return "; ".join(
        f"{item['method']}={normalize_confidence(item.get('confidence'))}:{item.get('score_a')}:{item.get('score_b')}"
        for item in attempts
    )


def should_stop_retry(result: dict) -> bool:
    """Stop retrying only when confidence is high and both scores exist."""
    confidence = normalize_confidence(result.get("confidence"))
    scores_ok = result.get("score_a") is not None and result.get("score_b") is not None
    return confidence == "high" and scores_ok


# ── Gemini API ────────────────────────────────────────────────────────────────

def call_gemini(
    image_bytes: bytes,
    api_key: str,
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
    retry_base_delay: float = 10.0,
) -> dict:
    """Call Gemini API with an image and return parsed JSON response."""
    import urllib.request
    import urllib.error

    url = API_URL.format(model=model, key=api_key)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                    {"text": SCORE_PROMPT_SINGLE},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 256,
        },
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 or e.code >= 500:
                wait = retry_base_delay * attempt
                print(f"    HTTP {e.code}, retry {attempt}/{max_retries} in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"    HTTP {e.code}: {err_body}", file=sys.stderr)
                raise
        except Exception as e:
            wait = retry_base_delay * attempt
            print(f"    Error: {e}, retry {attempt}/{max_retries} in {wait:.0f}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} retries")


def parse_response(data: dict) -> dict:
    """Extract score JSON from Gemini response."""
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return {"score_a": None, "score_b": None, "confidence": "low", "note": "empty response"}

    # strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {"score_a": None, "score_b": None, "confidence": "low", "note": f"parse error: {text[:100]}"}

    # normalize
    for key in ("score_a", "score_b"):
        val = result.get(key)
        if val is not None:
            try:
                result[key] = int(val)
            except (ValueError, TypeError):
                result[key] = None
    result["confidence"] = normalize_confidence(result.get("confidence"))
    result.setdefault("note", "")
    return result


def call_gemini_for_bytes(
    image_bytes: bytes,
    api_key: str,
    model: str,
    max_retries: int,
) -> dict:
    """Call Gemini and parse the returned scoreboard JSON."""
    raw = call_gemini(image_bytes, api_key, model=model, max_retries=max_retries)
    return parse_response(raw)


def score_image_path(
    image_path: Path,
    api_key: str,
    model: str,
    max_retries: int,
    resize_width: int | None,
) -> tuple[dict, list[dict]]:
    """Score a single image file with one Gemini call."""
    image_bytes = load_and_resize(str(image_path), resize_width)
    parsed = call_gemini_for_bytes(image_bytes, api_key, model, max_retries)
    parsed["method"] = "image"
    return parsed, [parsed]


def score_video_path(
    video_path: Path,
    api_key: str,
    model: str,
    max_retries: int,
    resize_width: int | None,
    n_frames: int,
    sigma_clip_k: float,
    sigma_clip_iter: int,
) -> tuple[dict, list[dict]]:
    """Score a video by trying dominant_cluster, sigma_clip, median, mean, then max."""
    frames = extract_frames(str(video_path), n_frames, resize_width)
    if len(frames) < 3:
        raise ValueError("need at least 3 frames to build a composite")

    attempts: list[dict] = []
    best: dict | None = None

    for method_name, method_func in FALLBACK_METHODS:
        print(f"\n    trying {method_name}...", end=" ", flush=True)
        if method_name == "dominant_cluster":
            composite = method_func(frames)
        elif method_name == "sigma_clip":
            composite = method_func(frames, sigma=sigma_clip_k, iterations=sigma_clip_iter)
        else:
            composite = method_func(frames)

        image_bytes = image_to_jpeg_bytes(composite)
        parsed = call_gemini_for_bytes(image_bytes, api_key, model, max_retries)
        parsed["method"] = method_name
        attempts.append(parsed)
        print(f"→ {parsed['score_a']} : {parsed['score_b']}  ({parsed['confidence']})")

        if best is None or candidate_rank(parsed) > candidate_rank(best):
            best = parsed

        if should_stop_retry(parsed):
            break

    if best is None:
        raise RuntimeError("no valid Gemini responses returned")
    return best, attempts


def source_label(path: Path, root: Path | None) -> str:
    """Render a stable label for CSV output."""
    if root is None:
        return path.name
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Read badminton scoreboards via Gemini Flash 2.5")
    parser.add_argument("input", help="Video file, image file, or a folder of videos")
    parser.add_argument("-o", "--output", default="scores.csv", help="Output CSV path (default: scores.csv)")
    parser.add_argument("--resize", type=int, default=None,
                        help="Max image width in pixels (preserves aspect ratio)")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="Delay in seconds between API calls (default: 4.0)")
    parser.add_argument("--n-frames", type=int, default=30,
                        help="Number of frames to sample from each video (default: 30)")
    parser.add_argument("--sigma-clip-k", type=float, default=2.0,
                        help="Sigma multiplier for sigma_clip (default: 2.0)")
    parser.add_argument("--sigma-clip-iter", type=int, default=3,
                        help="Iterations for sigma_clip (default: 3)")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Gemini model name (default: gemini-2.5-flash)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries on transient errors (default: 3)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY environment variable not set.\n"
                 "  PowerShell: $env:GEMINI_API_KEY = \"your-key\"\n"
                 "  Bash:       export GEMINI_API_KEY=\"your-key\"")

    inputs = collect_inputs(args.input)
    input_root = Path(args.input) if Path(args.input).is_dir() else None
    print(f"Found {len(inputs)} item(s), model={args.model}, delay={args.delay}s")
    if args.resize:
        print(f"  Resizing to max width {args.resize}px")

    results = []
    for i, input_path in enumerate(inputs):
        label = f"[{i + 1}/{len(inputs)}]"
        display_name = source_label(input_path, input_root)
        print(f"{label} {display_name}...", end=" ", flush=True)

        try:
            if input_path.suffix.lower() in SUPPORTED_VIDEO_EXTS:
                parsed, attempts = score_video_path(
                    video_path=input_path,
                    api_key=api_key,
                    model=args.model,
                    max_retries=args.max_retries,
                    resize_width=args.resize,
                    n_frames=args.n_frames,
                    sigma_clip_k=args.sigma_clip_k,
                    sigma_clip_iter=args.sigma_clip_iter,
                )
            else:
                parsed, attempts = score_image_path(
                    image_path=input_path,
                    api_key=api_key,
                    model=args.model,
                    max_retries=args.max_retries,
                    resize_width=args.resize,
                )

            print(f"→ {parsed['score_a']} : {parsed['score_b']}  ({parsed['confidence']}, {parsed.get('method')})")

            results.append({
                "file": display_name,
                "score_a": parsed["score_a"],
                "score_b": parsed["score_b"],
                "confidence": parsed["confidence"],
                "method": parsed.get("method", ""),
                "attempts": format_attempts(attempts),
                "note": parsed["note"],
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "file": display_name,
                "score_a": None,
                "score_b": None,
                "confidence": "error",
                "method": "",
                "attempts": "",
                "note": str(e)[:200],
            })

        # rate limit: wait between items
        if i < len(inputs) - 1:
            time.sleep(args.delay)

    # write CSV
    out_path = args.output
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "score_a", "score_b", "confidence", "method", "attempts", "note"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! {len(results)} results → {out_path}")

    # summary
    high = sum(1 for r in results if r["confidence"] == "high")
    med = sum(1 for r in results if r["confidence"] == "medium")
    low = sum(1 for r in results if r["confidence"] == "low")
    err = sum(1 for r in results if r["confidence"] == "error")
    print(f"  Confidence: {high} high, {med} medium, {low} low, {err} error")


if __name__ == "__main__":
    main()
