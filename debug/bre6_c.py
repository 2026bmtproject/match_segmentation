"""
bre6_c.py

讀取影片，計算每一幀與前一幀的 Difference Score，
使用 Gaussian Mixture Model (GMM) 自動找低動態 / 高動態分界，
將低於分界的連續片段輸出成 CSV。

輸出格式參考 test3_segments.csv：
    Segment_ID, Start_Frame, End_Frame, Start_Sec, End_Sec, Duration_Sec

規則：
    - 低於分界的幀視為候選片段
    - 兩個候選片段的間隔小於 3 frame 時，合併為同一片段
    - 片段最小長度為 1 秒，短於此長度的片段會被丟棄

用法:
    python bre6_c.py test3.mp4 output.csv

若未提供參數:
    - input.mp4 會使用目前目錄第一個 .mp4
    - output.csv 會命名為 <影片檔名>_segments.csv
"""

from __future__ import annotations

import csv
import glob
import os
import sys

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


MIN_SEGMENT_SEC = 1.0
MERGE_GAP_FRAMES = 3


def pick_default_video() -> str:
    videos = sorted(glob.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError("找不到 .mp4 檔案，請提供影片路徑")
    return videos[0]


def find_threshold_gmm(scores: np.ndarray) -> float:
    """使用 2-component GMM 在 log10(score) 上估計分界點。"""
    if scores.size == 0:
        raise ValueError("scores 為空")

    log_scores = np.log10(np.maximum(scores.astype(float), 1.0))
    x = log_scores.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
    )
    gmm.fit(x)

    means = np.asarray(gmm.means_).ravel()
    variances = np.asarray(gmm.covariances_).ravel()
    weights = np.asarray(gmm.weights_).ravel()

    order = np.argsort(means)
    m1, m2 = means[order]
    v1, v2 = variances[order]
    w1, w2 = weights[order]

    v1 = max(float(v1), 1e-12)
    v2 = max(float(v2), 1e-12)

    s1 = np.sqrt(v1)
    s2 = np.sqrt(v2)

    a = 1.0 / (2.0 * v1) - 1.0 / (2.0 * v2)
    b = m2 / v2 - m1 / v1
    c = (m1**2) / (2.0 * v1) - (m2**2) / (2.0 * v2) + np.log((s2 * w1) / (s1 * w2))

    roots = np.roots([a, b, c])
    real_roots = np.real(roots[np.isreal(roots)])
    between = real_roots[(real_roots > m1) & (real_roots < m2)]

    if between.size > 0:
        thresh_log = float(between[0])
    else:
        # 若沒有解析解落在兩群中間，退回兩均值中點，避免崩潰。
        thresh_log = float((m1 + m2) / 2.0)

    return float(10.0**thresh_log)


def compute_frame_diff(video_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    ok, prev_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("無法讀取第一幀")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diffs = [0.0]
    times = [0.0]

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_no += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        score = float(np.sum(diff))
        diffs.append(score)
        times.append(frame_no / fps)
        prev_gray = gray

        if frame_no % 300 == 0:
            print(f"  已處理 {frame_no} 幀...", end="\r")

    cap.release()
    print()
    return np.array(diffs, dtype=float), np.array(times, dtype=float), float(fps)


def merge_close_segments(segments: list[tuple[int, int]], gap_frames: int) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged = [segments[0]]
    for start_frame, end_frame in segments[1:]:
        prev_start, prev_end = merged[-1]
        gap = start_frame - prev_end - 1
        if gap < gap_frames:
            merged[-1] = (prev_start, max(prev_end, end_frame))
        else:
            merged.append((start_frame, end_frame))
    return merged


def build_segments_from_mask(is_low: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start_frame: int | None = None

    for frame_idx, low in enumerate(is_low):
        if low:
            if start_frame is None:
                start_frame = frame_idx
        elif start_frame is not None:
            segments.append((start_frame, frame_idx - 1))
            start_frame = None

    if start_frame is not None:
        segments.append((start_frame, len(is_low) - 1))

    return segments


def filter_short_segments(
    segments: list[tuple[int, int]],
    fps: float,
    min_segment_sec: float,
) -> list[tuple[int, int]]:
    return [
        (start_frame, end_frame)
        for start_frame, end_frame in segments
        if ((end_frame - start_frame) / fps) >= min_segment_sec
    ]


def write_segments_csv(
    output_csv: str,
    segments: list[tuple[int, int]],
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
        ])

        for segment_id, (start_frame, end_frame) in enumerate(segments, start=1):
            start_sec = start_frame / fps
            end_sec = end_frame / fps
            duration_sec = end_sec - start_sec
            writer.writerow([
                segment_id,
                start_frame,
                end_frame,
                f"{start_sec:.3f}",
                f"{end_sec:.3f}",
                f"{duration_sec:.3f}",
            ])


def main() -> None:
    video_path = sys.argv[1] if len(sys.argv) > 1 else pick_default_video()
    output_csv = (
        sys.argv[2]
        if len(sys.argv) > 2
        else f"{os.path.splitext(os.path.basename(video_path))[0]}_segments.csv"
    )

    print("=" * 60)
    print("bre6_c: FrameDiff + GMM 分界 + 片段輸出")
    print("=" * 60)
    print(f"影片: {video_path}")
    print(f"輸出: {output_csv}")

    scores, times, fps = compute_frame_diff(video_path)
    threshold = find_threshold_gmm(scores)

    is_low = scores < threshold
    if is_low.size > 0:
        is_low[0] = False

    raw_segments = build_segments_from_mask(is_low)
    merged_segments = merge_close_segments(raw_segments, MERGE_GAP_FRAMES)
    filtered_segments = filter_short_segments(merged_segments, fps, MIN_SEGMENT_SEC)

    write_segments_csv(output_csv, filtered_segments, fps)

    low_frames = int(np.sum(is_low))
    duration_sec = float(times[-1]) if times.size else 0.0

    print("-" * 60)
    print(f"FPS: {fps:.3f}")
    print(f"總幀數: {scores.size}")
    print(f"總時長: {duration_sec/60:.1f} 分鐘")
    print(f"GMM 分界值: {threshold:.2f}")
    print(f"低於分界: {low_frames} 幀 ({low_frames / max(scores.size, 1) * 100:.1f}%)")
    print(f"原始片段數: {len(raw_segments)}")
    print(f"合併後片段數: {len(merged_segments)}")
    print(f"保留片段數: {len(filtered_segments)}")
    print("完成")


if __name__ == "__main__":
    main()