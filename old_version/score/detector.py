"""Scoreboard detection: locate the scoreboard region using multi-frame VLM consensus.

Design:
  • Sample frames are interleaved into rounds of DETECT_BATCH_SIZE (3-5 each),
    so every round spans the full video rather than a single contiguous section.
  • Each round sends all images in one API call; the model returns a single BBox
    for the scoreboard that is consistently visible at a fixed position across
    all images in that round.
  • BBoxes from all rounds are clustered and averaged into the final consensus.

Standalone test (saves crops + annotated frames for visual verification):
    python detector.py <video_path>
    python detector.py <video_path> --n-segments 120 --output-dir detect_debug
    python detector.py <video_path> --n-samples 15 --batch-size 3
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
from google import genai
from google.genai import types

from config import (
    DETECT_BATCH_SIZE,
    DETECT_BBOX_MARGIN,
    DETECT_SAMPLE_MIN,
    DETECT_SAMPLE_PCT,
)
from models import BBox
from prompts import DETECT_BATCH_PROMPT


# ── public utility ────────────────────────────────────────────────────────────

def detect_sample_count(n_segments: int) -> int:
    """Return the number of sample frames to extract given a segment count."""
    return max(DETECT_SAMPLE_MIN, int(n_segments * DETECT_SAMPLE_PCT))


# ── abstract interface ────────────────────────────────────────────────────────

class ScoreboardDetector(ABC):
    @abstractmethod
    async def detect(self, sample_frames: list[tuple[str, bytes]]) -> list[BBox]:
        """Return 0, 1, or 2 consensus BBoxes (0 = detection failed)."""


# ── helpers ───────────────────────────────────────────────────────────────────

def _interleave_into_rounds(
    frames: list[tuple[str, bytes]],
    batch_size: int,
) -> list[list[tuple[str, bytes]]]:
    """Distribute frames into rounds round-robin so each round spans the full video.

    n_rounds = n // batch_size (at least 1).  Round-robin assignment ensures
    consecutive frames in the input (which are evenly spaced in time) end up in
    different rounds, so every round samples diverse moments.
    """
    n = len(frames)
    if n < 2:
        return []
    n_rounds = max(1, n // batch_size)
    rounds: list[list] = [[] for _ in range(n_rounds)]
    for i, frame in enumerate(frames):
        rounds[i % n_rounds].append(frame)
    return [r for r in rounds if len(r) >= 2]


def _cluster_bboxes(detections: list[BBox], gap_threshold: float = 0.3) -> list[list[BBox]]:
    """Split detections into 1 or 2 clusters by x-centre gap."""
    if len(detections) < 2:
        return [detections] if detections else []
    paired = sorted(
        ((b.x1 + b.x2) / 2, b.y1, idx, b)
        for idx, b in enumerate(detections)
    )
    max_gap, split_idx = max(
        ((paired[i + 1][0] - paired[i][0], i) for i in range(len(paired) - 1)),
        key=lambda x: x[0],
    )
    if max_gap > gap_threshold:
        return [
            [b for _, _, _, b in paired[: split_idx + 1]],
            [b for _, _, _, b in paired[split_idx + 1 :]],
        ]
    return [[b for _, _, _, b in paired]]


def _mean_bbox(cluster: list[BBox]) -> BBox:
    n = len(cluster)
    return BBox(
        x1=sum(b.x1 for b in cluster) / n,
        y1=sum(b.y1 for b in cluster) / n,
        x2=sum(b.x2 for b in cluster) / n,
        y2=sum(b.y2 for b in cluster) / n,
    )


def _strip_markdown(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return text


def sample_frames_from_video(
    video_path: Path,
    n_frames: int,
) -> list[tuple[str, bytes]]:
    """Extract n_frames evenly spaced frames from a video (no crop applied)."""
    from frame_extractor import extract_raw_frame

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if total <= 0:
        raise ValueError(f"無法讀取影片幀數：{video_path}")

    positions = [int(total * (i + 0.5) / n_frames) for i in range(n_frames)]
    frames: list[tuple[str, bytes]] = []
    for pos in positions:
        data = extract_raw_frame(video_path, pos)
        if data:
            sec = pos / fps
            frames.append((f"f{pos:06d}_{sec:.1f}s", data))
    return frames


# ── Gemini implementation ─────────────────────────────────────────────────────

class GeminiDetector(ScoreboardDetector):
    def __init__(self, client: genai.Client, sem: asyncio.Semaphore, model: str):
        self.client = client
        self.sem    = sem
        self.model  = model

    async def detect(self, sample_frames: list[tuple[str, bytes]]) -> list[BBox]:
        rounds = _interleave_into_rounds(sample_frames, DETECT_BATCH_SIZE)
        n_frames = len(sample_frames)
        n_rounds = len(rounds)
        sizes = [len(r) for r in rounds]
        print(
            f"\n記分板位置偵測：{n_frames} 幀 → {n_rounds} 輪"
            f"（每輪 {min(sizes)}–{max(sizes)} 幀）"
        )

        round_bboxes: list[BBox] = []
        for idx, round_frames in enumerate(rounds, 1):
            bbox = await self._detect_round(idx, round_frames)
            if bbox:
                round_bboxes.append(bbox)

        if not round_bboxes:
            print("  所有輪次均未偵測到記分板，使用全幀模式。")
            return []

        print(f"  有效輪次：{len(round_bboxes)}/{n_rounds}")
        clusters = _cluster_bboxes(round_bboxes)
        boxes = [_mean_bbox(c).with_margin() for c in clusters]

        margin_pct = f"{DETECT_BBOX_MARGIN * 100:.0f}%"
        if len(boxes) == 2:
            print("  偵測到兩個記分板：")
            for i, b in enumerate(boxes, 1):
                print(
                    f"    記分板 {i}（含 {margin_pct} 邊距）: "
                    f"x=[{b.x1:.3f}, {b.x2:.3f}]  y=[{b.y1:.3f}, {b.y2:.3f}]"
                )
        else:
            b = boxes[0]
            print(
                f"  共識位置（含 {margin_pct} 邊距）: "
                f"x=[{b.x1:.3f}, {b.x2:.3f}]  y=[{b.y1:.3f}, {b.y2:.3f}]"
            )
        return boxes

    async def _detect_round(
        self,
        round_idx: int,
        round_frames: list[tuple[str, bytes]],
    ) -> Optional[BBox]:
        n = len(round_frames)
        labels = [lbl for lbl, _ in round_frames]
        print(f"  [輪次 {round_idx}/{round_idx}] {n} 幀：{labels[0]} … {labels[-1]} ...")

        contents: list = []
        for i, (_, jpeg_bytes) in enumerate(round_frames, start=1):
            contents.append(f"[圖 {i}]")
            contents.append(types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"))
        contents.append(DETECT_BATCH_PROMPT.format(n=n))

        async with self.sem:
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        **{"response_mime_type": "application/json", "temperature": 0}
                    ),
                )
                await asyncio.sleep(1)
                text = _strip_markdown(response.text or "")
                parsed = json.loads(text)
                x1   = parsed.get("x1")
                y1   = parsed.get("y1")
                x2   = parsed.get("x2")
                y2   = parsed.get("y2")
                conf = parsed.get("confidence", "low")
                cnt  = parsed.get("consistent_count", 0)

                if all(v is not None for v in (x1, y1, x2, y2)) and conf != "low":
                    bbox = BBox(float(x1), float(y1), float(x2), float(y2))
                    if bbox.is_valid():
                        print(
                            f"    → x=[{x1:.3f}, {x2:.3f}]  y=[{y1:.3f}, {y2:.3f}]  "
                            f"[{conf}]  一致：{cnt}/{n} 幀"
                        )
                        return bbox
                    print(f"    → 座標無效，跳過")
                else:
                    print(f"    → 未找到穩定記分板 [{conf}]")
                return None
            except Exception as e:
                print(f"  [輪次 {round_idx} 錯誤] {e}")
                return None


# ── standalone test ───────────────────────────────────────────────────────────

def _save_debug_outputs(
    out_dir: Path,
    frames: list[tuple[str, bytes]],
    bboxes: list[BBox],
    round_results: list[Optional[dict]],
) -> None:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (label, jpeg_bytes) in enumerate(frames, 1):
        (out_dir / f"frame_{i:02d}__{label}.jpg").write_bytes(jpeg_bytes)

    if not bboxes:
        print(f"未偵測到記分板，已儲存 {len(frames)} 張輸入幀至 {out_dir}/")
        return

    for i, (label, jpeg_bytes) in enumerate(frames, 1):
        arr   = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w  = frame.shape[:2]

        vis = frame.copy()
        for bi, bbox in enumerate(bboxes):
            color = (0, 255, 0) if bi == 0 else (0, 128, 255)
            cv2.rectangle(
                vis,
                (int(bbox.x1 * w), int(bbox.y1 * h)),
                (int(bbox.x2 * w), int(bbox.y2 * h)),
                color, 3,
            )
            label_txt = f"bbox{bi + 1}" if len(bboxes) > 1 else "bbox"
            cv2.putText(
                vis, label_txt,
                (int(bbox.x1 * w) + 4, int(bbox.y1 * h) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
        (out_dir / f"vis_{i:02d}__{label}.jpg").write_bytes(
            cv2.imencode(".jpg", vis)[1].tobytes()
        )

        for bi, bbox in enumerate(bboxes):
            crop   = bbox.crop_frame(frame)
            suffix = f"_box{bi + 1}" if len(bboxes) > 1 else ""
            ok, buf = cv2.imencode(".jpg", crop)
            if ok:
                (out_dir / f"crop_{i:02d}__{label}{suffix}.jpg").write_bytes(
                    buf.tobytes()
                )

    (out_dir / "consensus_bbox.json").write_text(
        json.dumps([b.to_dict() for b in bboxes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if round_results:
        (out_dir / "round_results.json").write_text(
            json.dumps(round_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(
        f"\n已儲存至 {out_dir}/：\n"
        f"  frame_*.jpg  — {len(frames)} 張輸入幀\n"
        f"  vis_*.jpg    — 標記記分板框線\n"
        f"  crop_*.jpg   — 裁切後的記分板區域\n"
        f"  consensus_bbox.json\n"
        f"  round_results.json"
    )
    print(f"\n共識 BBox：")
    for b in bboxes:
        print(f"  {b.to_dict()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="記分板位置偵測獨立測試工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="影片路徑")
    parser.add_argument(
        "--n-segments", type=int, default=None,
        help="模擬的片段數量，用於計算取樣幀數（n_segments × DETECT_SAMPLE_PCT）",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="直接指定取樣幀數（優先於 --n-segments）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DETECT_BATCH_SIZE,
        help="每輪送給模型的圖片數（3-5）",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("detect_debug"),
        help="除錯輸出資料夾",
    )
    parser.add_argument("--model", default=None, help="Gemini 模型名稱")
    args = parser.parse_args()

    from config import API_KEY_ENV, MODEL

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"請先設定環境變數 {API_KEY_ENV}")
        sys.exit(1)

    video_path: Path = args.video
    if not video_path.exists():
        print(f"找不到影片：{video_path}")
        sys.exit(1)

    # Determine sample count
    if args.n_samples:
        n_samples = args.n_samples
    elif args.n_segments:
        n_samples = detect_sample_count(args.n_segments)
    else:
        cap   = cv2.VideoCapture(str(video_path))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration_min = (total / fps) / 60.0
        n_samples = max(DETECT_SAMPLE_MIN, int(duration_min * 3))

    batch_size  = max(3, min(5, args.batch_size))
    model_name  = args.model or MODEL

    print(f"影片：{video_path}")
    print(f"取樣幀數：{n_samples}，每輪：{batch_size}，模型：{model_name}")

    print("\n取樣中...")
    frames = sample_frames_from_video(video_path, n_samples)
    print(f"成功取得 {len(frames)} 幀")

    # Patch detector to also collect round-level raw responses
    client   = genai.Client(api_key=api_key)
    sem      = asyncio.Semaphore(1)
    detector = GeminiDetector(client, sem, model_name)

    round_results: list[Optional[dict]] = []

    original_detect_round = detector._detect_round

    async def _patched_round(round_idx, round_frames):
        bbox = await original_detect_round(round_idx, round_frames)
        round_results.append(
            {"round": round_idx, "n_frames": len(round_frames),
             "bbox": bbox.to_dict() if bbox else None}
        )
        return bbox

    detector._detect_round = _patched_round  # type: ignore[method-assign]

    bboxes = asyncio.run(detector.detect(frames))

    _save_debug_outputs(args.output_dir, frames, bboxes, round_results)
