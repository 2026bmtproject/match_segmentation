from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np

from config import DETECT_BBOX_MARGIN


@dataclasses.dataclass
class SegmentSource:
    label: str
    mode: str                   # "clip" or "csv"
    clip_path: Optional[Path]
    video_path: Optional[Path]
    start_frame: Optional[int]
    end_frame: Optional[int]
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None


@dataclasses.dataclass
class BBox:
    """Scoreboard bounding box in normalised coords [0, 1]."""
    x1: float
    y1: float
    x2: float
    y2: float

    def with_margin(self, margin: float = DETECT_BBOX_MARGIN) -> "BBox":
        return BBox(
            max(0.0, self.x1 - margin),
            max(0.0, self.y1 - margin),
            min(1.0, self.x2 + margin),
            min(1.0, self.y2 + margin),
        )

    def is_valid(self) -> bool:
        return (
            0.0 <= self.x1 < self.x2 <= 1.0
            and 0.0 <= self.y1 < self.y2 <= 1.0
            and (self.x2 - self.x1) > 0.01
            and (self.y2 - self.y1) > 0.01
        )

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        return frame[
            int(self.y1 * h) : int(self.y2 * h),
            int(self.x1 * w) : int(self.x2 * w),
        ]

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in dataclasses.asdict(self).items()}
