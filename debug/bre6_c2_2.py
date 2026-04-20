"""
bre6_c2_2.py

讀取影片，計算每一幀與前一幀的 MAD (Mean Absolute Difference) Score，
使用 Gaussian Mixture Model (GMM) 自動找低動態 / 高動態分界，
抓出候選片段後，對每段取 2 個代表幀，並與其他候選片段的代表幀做跨片段比對，輸出成 CSV。

輸出欄位：
    Segment_ID, Start_Frame, End_Frame, Start_Sec, End_Sec, Duration_Sec,
    Mid1_Frame, Mid2_Frame,
    Compared_Segments, Cross_Diff_Sum, Cross_Diff_Avg

規則：
    - 低於分界的幀視為候選片段
    - 兩個候選片段的間隔小於 4 frame 時，合併為同一片段
    - 合併後僅保留長度至少 0.5 秒的片段
    - 每個候選片段平分成兩份，取每份的中間 frame
    - 每個候選片段的 2 個代表 frame，會和其他候選片段的 2 個代表 frame 進行跨片段比對
    - 所有 MAD 分數會四捨五入到整數位

用法:
    python bre6_c2_2.py input.mp4 output.csv
    python bre6_c2_2.py test2.mp4 test2_c2_2_segments.csv --diff-csv test2_bre6.csv

若未提供參數:
    - input.mp4 會使用目前目錄第一個 .mp4
    - output.csv 會命名為 <影片檔名>_c2_2_segments.csv
    - --diff-csv 可省略；若提供會直接讀取 CSV 的 Difference_Score / Time_Sec，
      跳過第一步逐幀計算 diff
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from debug.bre6_shared import (
    build_segments_from_mask,
    collect_required_frames,
    compute_cross_segment_scores,
    compute_frame_diff,
    find_threshold_gmm,
    filter_short_segments,
    load_frame_diff_csv,
    load_required_gray_frames,
    merge_close_segments,
    pick_default_video,
    update_progress,
)


MERGE_GAP_FRAMES = 2
MIN_SEGMENT_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FrameDiff(MAD) + GMM 分段，並做候選片段跨片段比對(每段2幀)",
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="輸入影片路徑（預設為目前目錄第一個 .mp4）",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default=None,
        help="輸出 CSV 路徑（預設 <影片名>_c2_2_segments.csv）",
    )
    parser.add_argument(
        "--diff-csv",
        dest="diff_csv",
        default=None,
        help="可選：既有 frame diff CSV 路徑，提供後可跳過逐幀計算 diff",
    )
    return parser.parse_args()


def write_segments_csv(
    output_csv: str,
    segments: list[tuple[int, int]],
    pairs: list[tuple[int, int]],
    cross_sums: list[int],
    cross_avgs: list[int],
    compared_segments: int,
    fps: float,
) -> None:
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Segment_ID",
            "Start_Frame",
            "End_Frame",
            "Start_Sec",
            "End_Sec",
            "Duration_Sec",
            "Mid1_Frame",
            "Mid2_Frame",
            "Compared_Segments",
            "Cross_Diff_Sum",
            "Cross_Diff_Avg",
        ])

        seg_total = max(len(segments), 1)
        start_time = time.time()
        update_progress("輸出 CSV", 0, seg_total, start_time)

        for idx, ((start_frame, end_frame), (m1, m2), cross_sum, cross_avg) in enumerate(
            zip(segments, pairs, cross_sums, cross_avgs),
            start=1,
        ):
            start_sec = start_frame / fps
            end_sec = end_frame / fps
            duration_sec = end_sec - start_sec

            writer.writerow([
                idx,
                start_frame,
                end_frame,
                f"{start_sec:.3f}",
                f"{end_sec:.3f}",
                f"{duration_sec:.3f}",
                m1,
                m2,
                compared_segments,
                cross_sum,
                cross_avg,
            ])

            if idx % 20 == 0 or idx == seg_total:
                update_progress("輸出 CSV", idx, seg_total, start_time)


def main() -> None:
    args = parse_args()

    video_path = args.video_path if args.video_path else pick_default_video()
    output_csv = (
        args.output_csv
        if args.output_csv
        else f"{os.path.splitext(os.path.basename(video_path))[0]}_c2_2_segments.csv"
    )

    print("=" * 60)
    print("bre6_c2_2: FrameDiff(MAD) + GMM + 候選片段跨片段比對(每段2幀)")
    print("=" * 60)
    print(f"影片: {video_path}")
    print(f"輸出: {output_csv}")
    if args.diff_csv:
        print(f"FrameDiff CSV: {args.diff_csv}")

    if args.diff_csv:
        scores, times, fps, processed_frames = load_frame_diff_csv(args.diff_csv)
    else:
        scores, times, fps, processed_frames = compute_frame_diff(video_path)

    threshold = find_threshold_gmm(scores)

    is_low = scores < threshold
    if is_low.size > 0:
        is_low[0] = False

    raw_segments = build_segments_from_mask(is_low)
    merged_segments = merge_close_segments(raw_segments, MERGE_GAP_FRAMES)
    candidate_segments = filter_short_segments(merged_segments, fps, MIN_SEGMENT_SECONDS)

    pairs, required_frames = collect_required_frames(candidate_segments)
    max_required = max(required_frames) if required_frames else 0
    frame_cache = load_required_gray_frames(video_path, required_frames, max_required)

    cross_sums, cross_avgs, compared_segments = compute_cross_segment_scores(pairs, frame_cache)

    write_segments_csv(
        output_csv,
        candidate_segments,
        pairs,
        cross_sums,
        cross_avgs,
        compared_segments,
        fps,
    )

    low_frames = int(np.sum(is_low))
    duration_sec = float(times[-1]) if times.size else 0.0

    print("-" * 60)
    print(f"FPS: {fps:.3f}")
    print(f"總幀數: {processed_frames}")
    print(f"總時長: {duration_sec/60:.1f} 分鐘")
    print(f"GMM 分界值: {threshold:.2f}")
    print(f"低於分界: {low_frames} 幀 ({low_frames / max(scores.size, 1) * 100:.1f}%)")
    print(f"原始片段數: {len(raw_segments)}")
    print(f"合併後片段數: {len(merged_segments)}")
    print(f"候選片段數(最短 {MIN_SEGMENT_SECONDS:.1f} 秒): {len(candidate_segments)}")
    print(f"代表幀快取數: {len(frame_cache)}")
    print("完成")


if __name__ == "__main__":
    main()
