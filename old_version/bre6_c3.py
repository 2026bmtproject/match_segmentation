"""
bre6_c3.py

讀取影片，先沿用 bre6_c2 的流程：
1) 計算每幀 FrameDiff，使用 2-component GMM 找低動態分界
2) 萃取候選片段並合併近距離片段
3) 對每段取 3 個代表幀，計算跨片段 Cross_Diff_Avg

最後再使用 plot_test2_c2_segments 的 3-component GMM 演算法，
在 Cross_Diff_Avg 上估計門檻，將高於門檻的片段篩掉。

輸出欄位格式比照 test2_segments.csv：
    Segment_ID, Start_Frame, End_Frame, Start_Sec, End_Sec, Duration_Sec

用法:
    python bre6_c3.py test2.mp4 test2_c3_segments.csv

若未提供參數:
    - input.mp4 會使用目前目錄第一個 .mp4
    - output.csv 會命名為 <影片檔名>_c3_segments.csv
"""

from __future__ import annotations

import csv
import glob
import os
import sys
import time

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


MERGE_GAP_FRAMES = 3
BAR_WIDTH = 30
COMPARE_SIZE = (128, 72)


def pick_default_video() -> str:
    videos = sorted(glob.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError("找不到 .mp4 檔案，請提供影片路徑")
    return videos[0]


def update_progress(prefix: str, current: int, total: int, start_time: float) -> None:
    if total <= 0:
        return
    current = max(0, min(current, total))
    ratio = current / total
    filled = int(ratio * BAR_WIDTH)
    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
    elapsed = max(time.time() - start_time, 1e-9)
    fps_like = current / elapsed
    eta = (total - current) / max(fps_like, 1e-9)
    print(
        f"\r{prefix} [{bar}] {ratio * 100:6.2f}% ({current}/{total}) ETA {eta:6.1f}s",
        end="",
        flush=True,
    )
    if current >= total:
        print()


def find_frame_threshold_gmm(scores: np.ndarray) -> float:
    """使用 2-component GMM 在 log10(FrameDiff) 上估計低動態分界。"""
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
        thresh_log = float((m1 + m2) / 2.0)

    return float(10.0**thresh_log)


def find_cross_threshold_gmm(scores: np.ndarray) -> float:
    """沿用 plot_test2_c2_segments：3-component GMM，取低值群與中值群交點。"""
    if scores.size == 0:
        return 0.0
    if scores.size < 3:
        return float(np.max(scores))

    log_scores = np.log10(np.maximum(scores.astype(float), 1e-12))
    x = log_scores.reshape(-1, 1)

    try:
        gmm = GaussianMixture(
            n_components=3,
            covariance_type="full",
            random_state=0,
        )
        gmm.fit(x)
    except ValueError:
        return float(np.max(scores))

    means = np.asarray(gmm.means_, dtype=float).ravel()
    variances = np.asarray(gmm.covariances_, dtype=float).ravel()
    weights = np.asarray(gmm.weights_, dtype=float).ravel()

    order = np.argsort(means)
    m1, m2 = means[order][:2]
    v1, v2 = variances[order][:2]
    w1, w2 = weights[order][:2]

    v1 = max(float(v1), 1e-12)
    v2 = max(float(v2), 1e-12)

    s1 = np.sqrt(v1)
    s2 = np.sqrt(v2)

    a = 1.0 / (2.0 * v1) - 1.0 / (2.0 * v2)
    b = m2 / v2 - m1 / v1
    c = (m1**2) / (2.0 * v1) - (m2**2) / (2.0 * v2) + np.log((s2 * w1) / (s1 * w2))

    roots = np.roots([a, b, c])
    real_roots = np.real(roots[np.isreal(roots)])
    valid_roots = real_roots[(real_roots > m1) & (real_roots < m2)]

    if valid_roots.size > 0:
        thresh_log = float(valid_roots[0])
    else:
        thresh_log = float((m1 + m2) / 2.0)

    return float(10.0**thresh_log)


def compute_frame_diff(video_path: str) -> tuple[np.ndarray, np.ndarray, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = max(total_frames, 1)

    ok, prev_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("無法讀取第一幀")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diffs = [0.0]
    times = [0.0]

    start_time = time.time()
    update_progress("掃描 FrameDiff", 1, total_frames, start_time)

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

        if frame_no % 100 == 0:
            update_progress("掃描 FrameDiff", frame_no + 1, total_frames, start_time)

    cap.release()
    processed = frame_no + 1
    update_progress("掃描 FrameDiff", processed, total_frames, start_time)

    return np.array(diffs, dtype=float), np.array(times, dtype=float), float(fps), processed


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


def pick_three_middle_frames(start_frame: int, end_frame: int) -> tuple[int, int, int]:
    boundaries = np.linspace(start_frame, end_frame + 1, 4)
    mids: list[int] = []

    for i in range(3):
        left = int(np.floor(boundaries[i]))
        right_exclusive = int(np.floor(boundaries[i + 1]))

        left = max(left, start_frame)
        right_exclusive = min(max(right_exclusive, left + 1), end_frame + 1)

        part_mid = (left + (right_exclusive - 1)) // 2
        mids.append(int(part_mid))

    return mids[0], mids[1], mids[2]


def collect_required_frames(segments: list[tuple[int, int]]) -> tuple[list[tuple[int, int, int]], set[int]]:
    triplets: list[tuple[int, int, int]] = []
    required: set[int] = set()

    for start_frame, end_frame in segments:
        m1, m2, m3 = pick_three_middle_frames(start_frame, end_frame)
        triplets.append((m1, m2, m3))
        required.add(m1)
        required.add(m2)
        required.add(m3)

    return triplets, required


def load_required_gray_frames(
    video_path: str,
    required_frames: set[int],
    max_frame_index: int,
) -> dict[int, np.ndarray]:
    if not required_frames:
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")

    total_to_scan = max_frame_index + 1
    start_time = time.time()
    frame_cache: dict[int, np.ndarray] = {}

    update_progress("擷取代表幀", 0, total_to_scan, start_time)

    frame_idx = 0
    while frame_idx <= max_frame_index:
        grabbed = cap.grab()
        if not grabbed:
            break

        if frame_idx in required_frames:
            ok, frame = cap.retrieve()
            if not ok:
                cap.release()
                raise RuntimeError(f"無法取回幀: {frame_idx}")
            frame_cache[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_idx += 1
        if frame_idx % 200 == 0 or frame_idx == total_to_scan:
            update_progress("擷取代表幀", min(frame_idx, total_to_scan), total_to_scan, start_time)

    cap.release()
    update_progress("擷取代表幀", min(frame_idx, total_to_scan), total_to_scan, start_time)

    missing = required_frames.difference(frame_cache.keys())
    if missing:
        raise RuntimeError(f"有代表幀讀取失敗，缺少 {len(missing)} 個幀")

    return frame_cache


def build_segment_vectors(
    triplets: list[tuple[int, int, int]],
    frame_cache: dict[int, np.ndarray],
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []

    for m1, m2, m3 in triplets:
        f1 = cv2.resize(frame_cache[m1], COMPARE_SIZE, interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(frame_cache[m2], COMPARE_SIZE, interpolation=cv2.INTER_AREA)
        f3 = cv2.resize(frame_cache[m3], COMPARE_SIZE, interpolation=cv2.INTER_AREA)

        seg_vec = np.stack([
            f1.reshape(-1),
            f2.reshape(-1),
            f3.reshape(-1),
        ]).astype(np.int16)
        vectors.append(seg_vec)

    return vectors


def compute_cross_segment_scores(
    triplets: list[tuple[int, int, int]],
    frame_cache: dict[int, np.ndarray],
) -> tuple[list[float], list[float], int]:
    seg_count = len(triplets)
    if seg_count == 0:
        return [], [], 0
    if seg_count == 1:
        return [0.0], [0.0], 0

    vectors = build_segment_vectors(triplets, frame_cache)

    sums = np.zeros(seg_count, dtype=np.float64)
    compared_segments = seg_count - 1

    total_pairs = seg_count * (seg_count - 1) // 2
    done_pairs = 0
    start_time = time.time()
    update_progress("跨片段比對", 0, total_pairs, start_time)

    for i in range(seg_count):
        vi = vectors[i]
        for j in range(i + 1, seg_count):
            vj = vectors[j]
            pair_sum = float(np.abs(vi[:, None, :] - vj[None, :, :]).sum())
            sums[i] += pair_sum
            sums[j] += pair_sum

            done_pairs += 1
            if done_pairs % 20 == 0 or done_pairs == total_pairs:
                update_progress("跨片段比對", done_pairs, total_pairs, start_time)

    avgs = sums / (compared_segments * 9)
    return sums.tolist(), avgs.tolist(), compared_segments


def filter_segments_by_cross_avg(
    segments: list[tuple[int, int]],
    cross_avgs: list[float],
    threshold: float,
) -> list[tuple[int, int]]:
    filtered: list[tuple[int, int]] = []
    for seg, avg in zip(segments, cross_avgs):
        if avg <= threshold:
            filtered.append(seg)
    return filtered


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

        seg_total = max(len(segments), 1)
        start_time = time.time()
        update_progress("輸出 CSV", 0, seg_total, start_time)

        for idx, (start_frame, end_frame) in enumerate(segments, start=1):
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
            ])

            if idx % 20 == 0 or idx == seg_total:
                update_progress("輸出 CSV", idx, seg_total, start_time)


def main() -> None:
    video_path = sys.argv[1] if len(sys.argv) > 1 else pick_default_video()
    output_csv = (
        sys.argv[2]
        if len(sys.argv) > 2
        else f"{os.path.splitext(os.path.basename(video_path))[0]}_c3_segments.csv"
    )

    print("=" * 60)
    print("bre6_c3: FrameDiff + GMM + CrossDiff 過濾")
    print("=" * 60)
    print(f"影片: {video_path}")
    print(f"輸出: {output_csv}")

    scores, times, fps, processed_frames = compute_frame_diff(video_path)
    frame_threshold = find_frame_threshold_gmm(scores)

    is_low = scores < frame_threshold
    if is_low.size > 0:
        is_low[0] = False

    raw_segments = build_segments_from_mask(is_low)
    candidate_segments = merge_close_segments(raw_segments, MERGE_GAP_FRAMES)

    triplets, required_frames = collect_required_frames(candidate_segments)
    max_required = max(required_frames) if required_frames else 0
    frame_cache = load_required_gray_frames(video_path, required_frames, max_required)

    _cross_sums, cross_avgs, _compared_segments = compute_cross_segment_scores(triplets, frame_cache)

    cross_threshold = find_cross_threshold_gmm(np.asarray(cross_avgs, dtype=float))
    filtered_segments = filter_segments_by_cross_avg(candidate_segments, cross_avgs, cross_threshold)

    write_segments_csv(output_csv, filtered_segments, fps)

    low_frames = int(np.sum(is_low))
    duration_sec = float(times[-1]) if times.size else 0.0

    print("-" * 60)
    print(f"FPS: {fps:.3f}")
    print(f"總幀數: {processed_frames}")
    print(f"總時長: {duration_sec / 60:.1f} 分鐘")
    print(f"FrameDiff GMM 分界值: {frame_threshold:.2f}")
    print(f"低於分界: {low_frames} 幀 ({low_frames / max(scores.size, 1) * 100:.1f}%)")
    print(f"原始片段數: {len(raw_segments)}")
    print(f"候選片段數(合併後): {len(candidate_segments)}")
    print(f"Cross_Diff_Avg GMM 分界值: {cross_threshold:.2f}")
    print(f"最終保留片段數: {len(filtered_segments)}")
    print("完成")


if __name__ == "__main__":
    main()
