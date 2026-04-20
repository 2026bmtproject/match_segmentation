"""
bre6_save_first_rep.py

讀取影片，計算每一幀與前一幀的 MAD (Mean Absolute Difference) Score，
使用 Gaussian Mixture Model (GMM) 自動找低動態 / 高動態分界，
抓出候選片段後，將每個候選片段的第一個代表 frame 儲存成圖片到資料夾。

輸出內容：
    - 自動建立一個資料夾
    - 每個候選片段輸出一張圖片
    - 圖片檔名會包含 Segment_ID 與 frame 編號

規則：
    - 低於分界的幀視為候選片段
    - 兩個候選片段的間隔小於 5 frame 時，合併為同一片段
    - 合併後僅保留長度至少 2 秒的片段
    - 每個候選片段平分成兩份，取每份的中間 frame
    - 只儲存第一個代表 frame
    - --diff-csv 可省略；若提供會直接讀取 CSV 的 Difference_Score / Time_Sec，
      跳過第一步逐幀計算 diff

用法:
    python bre6_save_first_rep.py input.mp4
    python bre6_save_first_rep.py input.mp4 output_folder
    python bre6_save_first_rep.py test2.mp4 output_folder --diff-csv test2_bre6.csv

若未提供參數:
    - input.mp4 會使用目前目錄第一個 .mp4
    - output_folder 會命名為 <影片檔名>_candidate_first_frames
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from debug.bre6_shared import (
    build_segments_from_mask,
    compute_frame_diff,
    find_threshold_gmm,
    filter_short_segments,
    load_frame_diff_csv,
    merge_close_segments,
    pick_default_video,
    pick_two_middle_frames,
    update_progress,
)


MERGE_GAP_FRAMES = 5
MIN_SEGMENT_SECONDS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FrameDiff(MAD) + GMM 分段，並將每段第一個代表幀輸出成圖片",
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="輸入影片路徑（預設為目前目錄第一個 .mp4）",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="輸出資料夾（預設 <影片名>_candidate_first_frames）",
    )
    parser.add_argument(
        "--diff-csv",
        dest="diff_csv",
        default=None,
        help="可選：既有 frame diff CSV 路徑，提供後可跳過逐幀計算 diff",
    )
    return parser.parse_args()


def load_required_color_frames(
    video_path: str,
    required_frames: set[int],
) -> dict[int, np.ndarray]:
    if not required_frames:
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")

    ordered_frames = sorted(required_frames)
    total_to_scan = len(ordered_frames)
    min_frame = ordered_frames[0]
    max_frame = ordered_frames[-1]
    start_time = time.time()
    frame_cache: dict[int, np.ndarray] = {}

    update_progress("擷取代表幀", 0, total_to_scan, start_time)

    if min_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)

    current_frame = min_frame
    matched_count = 0
    while current_frame <= max_frame:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"無法讀取幀: {current_frame}")

        if current_frame in required_frames:
            frame_cache[current_frame] = frame
            matched_count += 1
            if matched_count % 20 == 0 or matched_count == total_to_scan:
                update_progress("擷取代表幀", matched_count, total_to_scan, start_time)

        current_frame += 1

    cap.release()
    update_progress("擷取代表幀", total_to_scan, total_to_scan, start_time)

    missing = required_frames.difference(frame_cache.keys())
    if missing:
        raise RuntimeError(f"有代表幀讀取失敗，缺少 {len(missing)} 個幀")

    return frame_cache


def save_frame_image(frame: np.ndarray, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ok = cv2.imwrite(output_path, frame)
    if not ok:
        raise RuntimeError(f"無法寫入圖片: {output_path}")


def main() -> None:
    args = parse_args()

    video_path = args.video_path if args.video_path else pick_default_video()
    output_dir = (
        args.output_dir
        if args.output_dir
        else f"{os.path.splitext(os.path.basename(video_path))[0]}_candidate_first_frames"
    )

    print("=" * 60)
    print("bre6_save_first_rep: FrameDiff(MAD) + GMM + 候選片段第一代表幀輸出")
    print("=" * 60)
    print(f"影片: {video_path}")
    print(f"輸出資料夾: {output_dir}")
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

    pairs = [pick_two_middle_frames(start_frame, end_frame) for start_frame, end_frame in candidate_segments]
    first_frames = {m1 for m1, _ in pairs}
    frame_cache = load_required_color_frames(video_path, first_frames)

    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0
    total_to_save = len(candidate_segments)
    start_time = time.time()
    update_progress("輸出圖片", 0, total_to_save, start_time)

    for idx, ((start_frame, end_frame), (m1, m2)) in enumerate(zip(candidate_segments, pairs), start=1):
        if m1 not in first_frames:
            continue

        start_sec = start_frame / fps
        end_sec = end_frame / fps
        duration_sec = end_sec - start_sec

        output_name = (
            f"segment_{idx:04d}_"
            f"f{m1:06d}_"
            f"s{start_sec:.3f}_e{end_sec:.3f}_d{duration_sec:.3f}.png"
        )
        output_path = os.path.join(output_dir, output_name)

        save_frame_image(frame_cache[m1], output_path)
        saved_count += 1

        if idx % 20 == 0 or idx == total_to_save:
            update_progress("輸出圖片", idx, total_to_save, start_time)

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
    print(f"輸出圖片數: {saved_count}")
    print("完成")


if __name__ == "__main__":
    main()