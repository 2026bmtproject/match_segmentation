from __future__ import annotations

import csv
import glob
import os
import time

import cv2
import numpy as np


BAR_WIDTH = 30
COMPARE_SIZE = (128, 72)


def round_to_int(value: float) -> int:
    return int(np.rint(value))


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


def find_threshold_gmm(scores: np.ndarray) -> float:
    """使用 2-component GMM 在 log10(score) 上估計分界點（NumPy 實作，無 sklearn 依賴）。"""
    if scores.size == 0:
        raise ValueError("scores 為空")

    log_scores = np.log10(np.maximum(scores.astype(float), 1.0))

    m1 = float(np.percentile(log_scores, 25))
    m2 = float(np.percentile(log_scores, 75))
    shared_var = float(np.var(log_scores))
    v1 = max(shared_var, 1e-6)
    v2 = max(shared_var, 1e-6)
    w1, w2 = 0.5, 0.5

    for _ in range(100):
        p1 = w1 * np.exp(-0.5 * ((log_scores - m1) ** 2) / v1) / np.sqrt(2.0 * np.pi * v1)
        p2 = w2 * np.exp(-0.5 * ((log_scores - m2) ** 2) / v2) / np.sqrt(2.0 * np.pi * v2)
        denom = np.maximum(p1 + p2, 1e-12)

        r1 = p1 / denom
        r2 = p2 / denom

        n1 = max(float(np.sum(r1)), 1e-9)
        n2 = max(float(np.sum(r2)), 1e-9)

        m1 = float(np.sum(r1 * log_scores) / n1)
        m2 = float(np.sum(r2 * log_scores) / n2)

        v1 = max(float(np.sum(r1 * (log_scores - m1) ** 2) / n1), 1e-9)
        v2 = max(float(np.sum(r2 * (log_scores - m2) ** 2) / n2), 1e-9)

        total = max(float(log_scores.size), 1e-9)
        w1 = n1 / total
        w2 = n2 / total

    means = np.array([m1, m2], dtype=float)
    variances = np.array([v1, v2], dtype=float)
    weights = np.array([w1, w2], dtype=float)

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
    diffs = [0]
    times = [0.0]

    start_time = time.time()
    update_progress("掃描 FrameDiff(MAD)", 1, total_frames, start_time)

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_no += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        score = round_to_int(float(np.mean(diff)))
        diffs.append(score)
        times.append(frame_no / fps)
        prev_gray = gray

        if frame_no % 100 == 0:
            update_progress("掃描 FrameDiff(MAD)", frame_no + 1, total_frames, start_time)

    cap.release()
    processed = frame_no + 1
    update_progress("掃描 FrameDiff(MAD)", processed, total_frames, start_time)

    return np.array(diffs, dtype=float), np.array(times, dtype=float), float(fps), processed


def load_frame_diff_csv(diff_csv_path: str, fallback_fps: float = 30.0) -> tuple[np.ndarray, np.ndarray, float, int]:
    """從既有 frame diff CSV 載入 Difference_Score 與時間，跳過逐幀掃描。"""
    if not os.path.exists(diff_csv_path):
        raise FileNotFoundError(f"找不到 frame diff CSV: {diff_csv_path}")

    scores: list[int] = []
    times: list[float] = []
    frames: list[int] = []

    with open(diff_csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("frame diff CSV 沒有欄位標頭")

        for row in reader:
            if "Difference_Score" not in row or row["Difference_Score"] in (None, ""):
                continue
            scores.append(round_to_int(float(row["Difference_Score"])))

            if "Time_Sec" in row and row["Time_Sec"] not in (None, ""):
                times.append(float(row["Time_Sec"]))
            elif "Frame" in row and row["Frame"] not in (None, ""):
                frame_idx = int(float(row["Frame"]))
                frames.append(frame_idx)
            else:
                frames.append(len(scores) - 1)

    if not scores:
        raise ValueError("frame diff CSV 沒有可用資料（缺少 Difference_Score）")

    if not times:
        times = [frame / max(fallback_fps, 1e-9) for frame in (frames if frames else range(len(scores)))]

    fps = fallback_fps
    if len(times) >= 2:
        deltas = np.diff(np.array(times, dtype=float))
        positive = deltas[deltas > 0]
        if positive.size > 0:
            fps = float(1.0 / np.median(positive))

    processed_frames = len(scores)
    return np.array(scores, dtype=float), np.array(times, dtype=float), float(fps), processed_frames


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


def filter_short_segments(
    segments: list[tuple[int, int]],
    fps: float,
    min_seconds: float,
) -> list[tuple[int, int]]:
    if not segments:
        return []

    min_frames = int(np.ceil(min_seconds * max(fps, 1e-9)))
    kept: list[tuple[int, int]] = []

    for start_frame, end_frame in segments:
        seg_frames = end_frame - start_frame + 1
        if seg_frames >= min_frames:
            kept.append((start_frame, end_frame))

    return kept


def pick_two_middle_frames(start_frame: int, end_frame: int) -> tuple[int, int]:
    """將片段分兩份，取各份中間幀。"""
    boundaries = np.linspace(start_frame, end_frame + 1, 3)
    mids: list[int] = []

    for i in range(2):
        left = int(np.floor(boundaries[i]))
        right_exclusive = int(np.floor(boundaries[i + 1]))

        left = max(left, start_frame)
        right_exclusive = min(max(right_exclusive, left + 1), end_frame + 1)

        part_mid = (left + (right_exclusive - 1)) // 2
        mids.append(int(part_mid))

    return mids[0], mids[1]


def collect_required_frames(segments: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], set[int]]:
    pairs: list[tuple[int, int]] = []
    required: set[int] = set()

    for start_frame, end_frame in segments:
        m1, m2 = pick_two_middle_frames(start_frame, end_frame)
        pairs.append((m1, m2))
        required.add(m1)
        required.add(m2)

    return pairs, required


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
    pairs: list[tuple[int, int]],
    frame_cache: dict[int, np.ndarray],
) -> list[np.ndarray]:
    """把每段 2 張代表幀縮圖後展平，供快速跨片段比對。"""
    vectors: list[np.ndarray] = []

    for m1, m2 in pairs:
        f1 = cv2.resize(frame_cache[m1], COMPARE_SIZE, interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(frame_cache[m2], COMPARE_SIZE, interpolation=cv2.INTER_AREA)

        seg_vec = np.stack([
            f1.reshape(-1),
            f2.reshape(-1),
        ]).astype(np.int16)
        vectors.append(seg_vec)

    return vectors


def compute_cross_segment_scores(
    pairs: list[tuple[int, int]],
    frame_cache: dict[int, np.ndarray],
) -> tuple[list[int], list[int], int]:
    """計算每段 2 幀和其他片段 2 幀的跨片段 MAD。"""
    seg_count = len(pairs)
    if seg_count == 0:
        return [], [], 0
    if seg_count == 1:
        return [0], [0], 0

    vectors = build_segment_vectors(pairs, frame_cache)

    sums = np.zeros(seg_count, dtype=np.float64)
    compared_segments = seg_count - 1

    total_pairs = seg_count * (seg_count - 1) // 2
    done_pairs = 0
    start_time = time.time()
    update_progress("跨片段比對(MAD)", 0, total_pairs, start_time)

    for i in range(seg_count):
        vi = vectors[i]
        for j in range(i + 1, seg_count):
            vj = vectors[j]
            mad_matrix = np.mean(np.abs(vi[:, None, :] - vj[None, :, :]), axis=2)
            pair_sum = float(np.sum(mad_matrix))
            sums[i] += pair_sum
            sums[j] += pair_sum

            done_pairs += 1
            if done_pairs % 20 == 0 or done_pairs == total_pairs:
                update_progress("跨片段比對(MAD)", done_pairs, total_pairs, start_time)

    avgs = sums / (compared_segments * 4)
    sums_int = [round_to_int(x) for x in sums.tolist()]
    avgs_int = [round_to_int(x) for x in avgs.tolist()]
    return sums_int, avgs_int, compared_segments