"""Binary-search gap filling: find intermediate scores between known score groups."""
from __future__ import annotations

from pathlib import Path

from config import MIN_BINARY_INTERVAL_SEC
from frame_extractor import extract_frame_at_sec
from grouper import group_by_score
from models import BBox
from scorer import ScoreReader


async def binary_search_gap(
    scorer: ScoreReader,
    video_path: Path,
    left_sec: float,
    right_sec: float,
    expected_low: tuple[int, int],
    expected_high: tuple[int, int],
    depth: int = 0,
    max_depth: int = 8,
    min_interval: float = MIN_BINARY_INTERVAL_SEC,
    crop_boxes: list[BBox] | None = None,
) -> list[dict]:
    """Recursively bisect [left_sec, right_sec] to find intermediate scores."""
    interval = right_sec - left_sec
    mid_sec  = (left_sec + right_sec) / 2
    label    = f"gap_{left_sec:.1f}_{right_sec:.1f}_d{depth}"

    if interval < min_interval or depth >= max_depth:
        return [{
            "clip":            label,
            "score_a":         None,
            "score_b":         None,
            "confidence":      "low",
            "note":            f"missing_rally: interval {interval:.1f}s",
            "source_time_sec": mid_sec,
            "gap_fill":        True,
        }]

    frame_bytes = extract_frame_at_sec(video_path, mid_sec, crop_boxes)
    if frame_bytes is None:
        return []

    sub_results = await scorer.query_batch([(label, frame_bytes)])
    r = sub_results[0]
    r["source_time_sec"] = mid_sec
    r["gap_fill"]        = True

    sa, sb = r.get("score_a"), r.get("score_b")
    low_total  = expected_low[0]  + expected_low[1]
    high_total = expected_high[0] + expected_high[1]
    if sa is not None and sb is not None:
        total = int(sa) + int(sb)
        if total < low_total or total > high_total:
            r["score_a"]    = None
            r["score_b"]    = None
            r["confidence"] = "low"
            r["note"]       = f"sanity fail: total={total} out of [{low_total},{high_total}]"
            sa = sb = None

    score = (int(sa), int(sb)) if sa is not None and sb is not None else None

    recurse_kwargs = dict(
        scorer=scorer, video_path=video_path,
        max_depth=max_depth, min_interval=min_interval, crop_boxes=crop_boxes,
    )

    if score is None or score == expected_low:
        return [r] + await binary_search_gap(
            **recurse_kwargs,
            left_sec=mid_sec, right_sec=right_sec,
            expected_low=expected_low, expected_high=expected_high,
            depth=depth + 1,
        )
    if score == expected_high:
        return [r] + await binary_search_gap(
            **recurse_kwargs,
            left_sec=left_sec, right_sec=mid_sec,
            expected_low=expected_low, expected_high=expected_high,
            depth=depth + 1,
        )

    left_results  = await binary_search_gap(
        **recurse_kwargs,
        left_sec=left_sec, right_sec=mid_sec,
        expected_low=expected_low, expected_high=score,
        depth=depth + 1,
    )
    right_results = await binary_search_gap(
        **recurse_kwargs,
        left_sec=mid_sec, right_sec=right_sec,
        expected_low=score, expected_high=expected_high,
        depth=depth + 1,
    )
    return left_results + [r] + right_results


async def fill_gaps(
    scorer: ScoreReader,
    raw_results: list[dict],
    csv_segments: list[dict],
    video_path: Path,
    min_interval: float = MIN_BINARY_INTERVAL_SEC,
    crop_boxes: list[BBox] | None = None,
) -> list[dict]:
    """Run binary search between any score groups that skip more than 1 point."""
    seg_by_label = {
        f"frame_{s['start_frame']}_{s['end_frame']}": s for s in csv_segments
    }
    valid_groups = [g for g in group_by_score(raw_results) if g["score_a"] is not None]

    extra: list[dict] = []
    for i in range(len(valid_groups) - 1):
        g_low, g_high = valid_groups[i], valid_groups[i + 1]
        a1, b1 = g_low["score_a"],  g_low["score_b"]
        a2, b2 = g_high["score_a"], g_high["score_b"]
        if (a2 - a1) + (b2 - b1) <= 1:
            continue

        low_segs  = [seg_by_label[c] for c in g_low["clips"]  if c in seg_by_label]
        high_segs = [seg_by_label[c] for c in g_high["clips"] if c in seg_by_label]
        if not low_segs or not high_segs:
            continue

        left_sec  = max(s["end_sec"]   for s in low_segs)
        right_sec = min(s["start_sec"] for s in high_segs)
        if left_sec >= right_sec:
            continue

        print(f"\n  [二分搜尋] ({a1},{b1})→({a2},{b2}) "
              f"時間區間 [{left_sec:.1f}, {right_sec:.1f}]s")
        found = await binary_search_gap(
            scorer=scorer,
            video_path=video_path,
            left_sec=left_sec,
            right_sec=right_sec,
            expected_low=(a1, b1),
            expected_high=(a2, b2),
            min_interval=min_interval,
            crop_boxes=crop_boxes,
        )
        extra.extend(found)

    return raw_results + extra
