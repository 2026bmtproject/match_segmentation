"""Video frame extraction utilities.

All functions accept an optional `crop_boxes` list; pass the BBoxes returned
by the scoreboard detector to crop-and-composite frames before JPEG encoding.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import COMPRESS_H, COMPRESS_W, JPEG_QUALITY
from models import BBox


def compress_frame_to_jpeg(
    frame: np.ndarray,
    crop_boxes: list[BBox] | None = None,
) -> bytes | None:
    if crop_boxes:
        if len(crop_boxes) == 1:
            frame = crop_boxes[0].crop_frame(frame)
            if frame.size == 0:
                return None
        else:
            crops = [b.crop_frame(frame) for b in crop_boxes]
            if any(c.size == 0 for c in crops):
                return None
            resized = []
            for c in crops:
                h, w = c.shape[:2]
                if h == 0:
                    return None
                scale = COMPRESS_H / h
                new_w = max(1, int(w * scale))
                resized.append(cv2.resize(c, (new_w, COMPRESS_H), interpolation=cv2.INTER_AREA))
            frame = np.hstack(resized)
    frame = cv2.resize(frame, (COMPRESS_W, COMPRESS_H), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return bytes(buf) if ok else None


def extract_frame_by_number(
    video_path: Path,
    frame_number: int,
    crop_boxes: list[BBox] | None = None,
) -> bytes | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number, 0))
    ok, frame = cap.read()
    cap.release()
    return compress_frame_to_jpeg(frame, crop_boxes) if ok else None


def extract_middle_frame_bytes(
    video_path: Path,
    crop_boxes: list[BBox] | None = None,
) -> bytes | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return extract_frame_by_number(video_path, max(total // 2, 0), crop_boxes)


def extract_frame_at_position(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    position: float,
    crop_boxes: list[BBox] | None = None,
) -> bytes | None:
    frame_number = (
        start_frame
        if end_frame <= start_frame
        else start_frame + int((end_frame - start_frame) * position)
    )
    return extract_frame_by_number(video_path, frame_number, crop_boxes)


def extract_frame_at_sec(
    video_path: Path,
    time_sec: float,
    crop_boxes: list[BBox] | None = None,
) -> bytes | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return extract_frame_by_number(video_path, int(time_sec * fps), crop_boxes)


def extract_raw_frame(video_path: Path, frame_number: int) -> bytes | None:
    """Extract without any crop — used for scoreboard detection sampling."""
    return extract_frame_by_number(video_path, frame_number, crop_boxes=None)
