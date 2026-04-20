#!/usr/bin/env python3
"""
將分段 CSV 正規化：
1) 片段間隔小於等於 gap_frames 視為同一片段合併
2) 每個片段最短長度補到 min_sec 秒

輸入欄位至少需要 Start_Frame, End_Frame。
若有 Start_Sec, End_Sec, Duration_Sec 會一併重算。

用法:
    python normalize_segments.py input.csv output.csv
    python normalize_segments.py input.csv output.csv --gap-frames 3 --min-sec 1.0
    python normalize_segments.py input.csv output.csv --fps 30
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass


@dataclass
class Segment:
    start: int
    end: int


def infer_fps(rows: list[dict[str, str]]) -> float:
    candidates: list[float] = []
    for row in rows:
        try:
            sf = int(float(row["Start_Frame"]))
            ef = int(float(row["End_Frame"]))
            ss = float(row.get("Start_Sec", ""))
            es = float(row.get("End_Sec", ""))
        except (TypeError, ValueError, KeyError):
            continue

        frame_len = ef - sf
        sec_len = es - ss
        if frame_len > 0 and sec_len > 0:
            candidates.append(frame_len / sec_len)

    if not candidates:
        return 30.0

    candidates.sort()
    return candidates[len(candidates) // 2]


def read_segments(path: str) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV 沒有標題列")
        fieldnames = reader.fieldnames
        if "Start_Frame" not in fieldnames or "End_Frame" not in fieldnames:
            raise ValueError("CSV 需要 Start_Frame 與 End_Frame 欄位")
        rows = [row for row in reader]
    return rows, fieldnames


def to_ranges(rows: list[dict[str, str]]) -> list[Segment]:
    out: list[Segment] = []
    for row in rows:
        try:
            s = int(float(row["Start_Frame"]))
            e = int(float(row["End_Frame"]))
        except (TypeError, ValueError, KeyError):
            continue
        if e < s:
            s, e = e, s
        out.append(Segment(start=s, end=e))
    out.sort(key=lambda x: (x.start, x.end))
    return out


def merge_with_gap(segments: list[Segment], gap_frames: int) -> list[Segment]:
    if not segments:
        return []

    merged: list[Segment] = [Segment(segments[0].start, segments[0].end)]
    for seg in segments[1:]:
        cur = merged[-1]
        gap = seg.start - cur.end - 1
        if gap <= gap_frames:
            cur.end = max(cur.end, seg.end)
        else:
            merged.append(Segment(seg.start, seg.end))
    return merged


def enforce_min_duration(segments: list[Segment], min_frames: int) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        length = seg.end - seg.start + 1
        if length < min_frames:
            seg = Segment(seg.start, seg.start + min_frames - 1)
        out.append(seg)
    return out


def write_output(
    path: str,
    fieldnames: list[str],
    segments: list[Segment],
    fps: float,
) -> None:
    base_fields = [
        "Segment_ID",
        "Start_Frame",
        "End_Frame",
        "Start_Sec",
        "End_Sec",
        "Duration_Sec",
    ]

    final_fields = fieldnames[:]
    for name in base_fields:
        if name not in final_fields:
            final_fields.append(name)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields)
        writer.writeheader()

        for i, seg in enumerate(segments, start=1):
            start_sec = seg.start / fps
            end_sec = seg.end / fps
            duration_sec = (seg.end - seg.start + 1) / fps

            row = {key: "" for key in final_fields}
            row["Segment_ID"] = str(i)
            row["Start_Frame"] = str(seg.start)
            row["End_Frame"] = str(seg.end)
            row["Start_Sec"] = f"{start_sec:.3f}"
            row["End_Sec"] = f"{end_sec:.3f}"
            row["Duration_Sec"] = f"{duration_sec:.3f}"
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="合併近距離片段並補齊最短時長")
    parser.add_argument("input_csv", help="輸入 CSV")
    parser.add_argument("output_csv", help="輸出 CSV")
    parser.add_argument(
        "--gap-frames",
        type=int,
        default=3,
        help="片段間可容忍間距（<= 此值就合併），預設 3",
    )
    parser.add_argument(
        "--min-sec",
        type=float,
        default=1.0,
        help="片段最小秒數，預設 1.0",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="影片 FPS（未提供時會由 CSV 估計，估不到則用 30）",
    )
    args = parser.parse_args()

    if args.gap_frames < 0:
        raise ValueError("--gap-frames 不能小於 0")
    if args.min_sec <= 0:
        raise ValueError("--min-sec 需大於 0")

    rows, fieldnames = read_segments(args.input_csv)
    fps = args.fps if args.fps and args.fps > 0 else infer_fps(rows)

    segments = to_ranges(rows)
    merged = merge_with_gap(segments, args.gap_frames)

    min_frames = max(1, int(round(args.min_sec * fps)))
    padded = enforce_min_duration(merged, min_frames)

    # 補長後可能跨到下一段，再做一次合併與補齊。
    merged_again = merge_with_gap(padded, args.gap_frames)
    final_segments = enforce_min_duration(merged_again, min_frames)

    write_output(args.output_csv, fieldnames, final_segments, fps)

    print(
        f"完成: 輸入 {len(segments)} 段 -> 輸出 {len(final_segments)} 段, "
        f"fps={fps:.3f}, min_frames={min_frames}, gap_frames={args.gap_frames}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
