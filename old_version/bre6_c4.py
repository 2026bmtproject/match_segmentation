"""
bre6_c4.py

讀取影片，流程：
1) 計算每幀 FrameDiff，使用 2-component GMM 找低動態分界
2) 萃取候選片段並合併近距離片段
3) 對每段取 3 個代表幀，計算 KNN 局部密度分數（KNN_Dist）
4) 使用 2-component GMM 在 KNN_Dist 上找分界，保留低密度距離的片段

相較於 bre6_c3 的 Cross_Diff_Avg（與所有片段的平均距離）：
  - Cross_Diff_Avg 的盲點：若非比賽畫面數量多且彼此相似（如演播室），
    它們也會形成低分群，無法被剔除。
  - KNN_Dist 的優勢：只看每個片段與其「最相似的 K 個鄰居」的距離。
    比賽畫面鏡位固定 → 彼此極相似 → KNN_Dist 極小（緊密群集）。
    即使非比賽畫面是多數，只要比賽畫面夠緊密，它們的 KNN_Dist
    就會明顯低於其他場景，GMM 能可靠地找到分界。

輸出欄位格式比照 test2_segments.csv：
    Segment_ID, Start_Frame, End_Frame, Start_Sec, End_Sec, Duration_Sec

用法:
    python bre6_c4.py test2.mp4 test2_c4_segments.csv

若已有 frame diff CSV，可跳過逐幀掃描：
    python bre6_c4.py test2.mp4 test2_c4_segments.csv --diff-csv test2_bre6.csv

若未提供參數:
    - input.mp4 會使用目前目錄第一個 .mp4
    - output.csv 會命名為 <影片檔名>_c4_segments.csv
"""

from __future__ import annotations

import argparse
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
COMPARE_SIZE = (128, 72)      # 代表幀縮放大小（寬, 高）
KNN_K = 5                     # KNN 鄰居數（候選片段很少時會自動調低）
MIN_SEGMENT_SEC = 2.0         # 片段最小長度（秒）


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 步驟 1：FrameDiff 掃描
# ---------------------------------------------------------------------------

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
        diffs.append(float(np.sum(diff)))
        times.append(frame_no / fps)
        prev_gray = gray

        if frame_no % 100 == 0:
            update_progress("掃描 FrameDiff", frame_no + 1, total_frames, start_time)

    cap.release()
    processed = frame_no + 1
    update_progress("掃描 FrameDiff", processed, total_frames, start_time)

    return np.array(diffs, dtype=float), np.array(times, dtype=float), float(fps), processed


def load_frame_diff_csv(diff_csv_path: str, fallback_fps: float = 30.0) -> tuple[np.ndarray, np.ndarray, float, int]:
    """從既有 frame diff CSV 載入 Difference_Score 與時間，跳過逐幀掃描。"""
    if not os.path.exists(diff_csv_path):
        raise FileNotFoundError(f"找不到 frame diff CSV: {diff_csv_path}")

    scores: list[float] = []
    times: list[float | None] = []
    frames: list[int | None] = []

    with open(diff_csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("frame diff CSV 沒有欄位標頭")

        for row_index, row in enumerate(reader):
            score_text = row.get("Difference_Score")
            if score_text in (None, ""):
                continue

            scores.append(float(score_text))

            time_text = row.get("Time_Sec")
            frame_text = row.get("Frame")

            times.append(float(time_text) if time_text not in (None, "") else None)
            if frame_text not in (None, ""):
                frames.append(int(float(frame_text)))
            else:
                frames.append(row_index)

    if not scores:
        raise ValueError("frame diff CSV 沒有可用資料（缺少 Difference_Score）")

    fps = fallback_fps
    known_times = [t for t in times if t is not None]
    if len(known_times) >= 2:
        deltas = np.diff(np.array(known_times, dtype=float))
        positive = deltas[deltas > 0]
        if positive.size > 0:
            fps = float(1.0 / np.median(positive))

    filled_times: list[float] = []
    for idx, time_value in enumerate(times):
        if time_value is not None:
            filled_times.append(float(time_value))
        else:
            frame_value = frames[idx]
            frame_idx = int(frame_value) if frame_value is not None else idx
            filled_times.append(frame_idx / max(fps, 1e-9))

    processed_frames = len(scores)
    return np.array(scores, dtype=float), np.array(filled_times, dtype=float), float(fps), processed_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FrameDiff GMM + KNN 局部密度過濾")
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
        help="輸出 CSV 路徑（預設 <影片名>_c4_segments.csv）",
    )
    parser.add_argument(
        "--diff-csv",
        dest="diff_csv",
        default=None,
        help="可選：既有 frame diff CSV 路徑，提供後可跳過逐幀計算 diff",
    )
    return parser.parse_args()


def find_frame_threshold_gmm(scores: np.ndarray) -> float:
    """使用 2-component GMM 在 log10(FrameDiff) 上估計低動態分界。"""
    if scores.size == 0:
        raise ValueError("scores 為空")

    log_scores = np.log10(np.maximum(scores.astype(float), 1.0))
    x = log_scores.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(x)

    means = np.asarray(gmm.means_).ravel()
    variances = np.asarray(gmm.covariances_).ravel()
    weights = np.asarray(gmm.weights_).ravel()

    order = np.argsort(means)
    m1, m2 = means[order]
    v1, v2 = np.maximum(variances[order], 1e-12)
    w1, w2 = weights[order]

    s1, s2 = np.sqrt(v1), np.sqrt(v2)
    a = 1.0 / (2.0 * v1) - 1.0 / (2.0 * v2)
    b = m2 / v2 - m1 / v1
    c = (m1 ** 2) / (2.0 * v1) - (m2 ** 2) / (2.0 * v2) + np.log((s2 * w1) / (s1 * w2))

    roots = np.roots([a, b, c])
    real_roots = np.real(roots[np.isreal(roots)])
    between = real_roots[(real_roots > m1) & (real_roots < m2)]

    thresh_log = float(between[0]) if between.size > 0 else float((m1 + m2) / 2.0)
    return float(10.0 ** thresh_log)


# ---------------------------------------------------------------------------
# 步驟 2：候選片段建構
# ---------------------------------------------------------------------------

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
        if start_frame - prev_end - 1 < gap_frames:
            merged[-1] = (prev_start, max(prev_end, end_frame))
        else:
            merged.append((start_frame, end_frame))
    return merged


def filter_segments_by_min_duration(
    segments: list[tuple[int, int]],
    fps: float,
    min_duration_sec: float,
) -> list[tuple[int, int]]:
    """
    依輸出定義的 Duration_Sec（(end-start)/fps）過濾過短片段。

    例如 30fps 且最小 2 秒時，需滿足 (end-start) >= 60。
    """
    if not segments:
        return []

    fps_safe = max(float(fps), 1e-9)
    min_frames_delta = min_duration_sec * fps_safe
    return [
        (start_frame, end_frame)
        for start_frame, end_frame in segments
        if (end_frame - start_frame) >= min_frames_delta
    ]


# ---------------------------------------------------------------------------
# 步驟 3：代表幀擷取
# ---------------------------------------------------------------------------

def pick_three_middle_frames(start_frame: int, end_frame: int) -> tuple[int, int, int]:
    boundaries = np.linspace(start_frame, end_frame + 1, 4)
    mids: list[int] = []
    for i in range(3):
        left = max(int(np.floor(boundaries[i])), start_frame)
        right_exclusive = min(max(int(np.floor(boundaries[i + 1])), left + 1), end_frame + 1)
        mids.append((left + right_exclusive - 1) // 2)
    return mids[0], mids[1], mids[2]


def collect_required_frames(
    segments: list[tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], set[int]]:
    triplets: list[tuple[int, int, int]] = []
    required: set[int] = set()

    for start_frame, end_frame in segments:
        m1, m2, m3 = pick_three_middle_frames(start_frame, end_frame)
        triplets.append((m1, m2, m3))
        required.update([m1, m2, m3])

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


# ---------------------------------------------------------------------------
# 步驟 4：KNN 局部密度過濾（核心改動）
# ---------------------------------------------------------------------------

def build_segment_features(
    triplets: list[tuple[int, int, int]],
    frame_cache: dict[int, np.ndarray],
) -> np.ndarray:
    """
    每個片段取 3 個代表幀的像素平均作為特徵向量。

    使用均值而非拼接的原因：
      - 降低單幀噪音，更能反映片段的「整體視覺外觀」。
      - 同一鏡位的比賽畫面均值會高度一致；
        不同場景（演播室、廣告）的均值則與比賽畫面有明顯差距。

    回傳 shape: (N, COMPARE_SIZE[0] * COMPARE_SIZE[1])，dtype float32
    """
    features: list[np.ndarray] = []
    for m1, m2, m3 in triplets:
        f1 = cv2.resize(frame_cache[m1], COMPARE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
        f2 = cv2.resize(frame_cache[m2], COMPARE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
        f3 = cv2.resize(frame_cache[m3], COMPARE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
        mean_frame = (f1 + f2 + f3) / 3.0
        features.append(mean_frame.ravel())
    return np.array(features, dtype=np.float32)


def compute_knn_density_scores(
    triplets: list[tuple[int, int, int]],
    frame_cache: dict[int, np.ndarray],
    k: int = KNN_K,
) -> list[float]:
    """
    對每個片段計算其與 K 個最近鄰的平均 L1 距離（KNN_Dist）。

    為何優於 Cross_Diff_Avg：
      - Cross_Diff_Avg 衡量「與所有片段的距離」→ 多數群天生分數低。
        若非比賽畫面為多數，它們的分數反而低，篩選失效。
      - KNN_Dist 衡量「局部鄰域緊密程度」→ 只要比賽畫面彼此相似，
        無論它們是多數或少數，局部密度都會高於視覺各異的非比賽畫面。

    演算法：
      1. 建立每個片段的特徵向量（3幀均值）
      2. 計算全對全 L1 距離矩陣（向量化，O(N²D)）
      3. 每個片段排除自身後取最小 K 個距離的平均
      4. 正規化（除以特徵維度）使數值與解析度無關
    """
    seg_count = len(triplets)
    if seg_count == 0:
        return []
    if seg_count == 1:
        return [0.0]

    features = build_segment_features(triplets, frame_cache)  # (N, D)
    N, D = features.shape
    actual_k = min(k, N - 1)

    print(f"\n[KNN密度] 片段數={N}, K={actual_k}, 特徵維度={D}")

    # 計算全對全距離矩陣（向量化）
    start_time = time.time()
    update_progress("KNN 距離矩陣", 0, N, start_time)

    dist_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        dist_matrix[i] = np.abs(features - features[i]).sum(axis=1)
        if (i + 1) % 10 == 0 or i + 1 == N:
            update_progress("KNN 距離矩陣", i + 1, N, start_time)

    # 正規化至每像素均差
    dist_matrix /= float(D)

    # 排除自身（設為 inf），取 K 最近鄰平均
    np.fill_diagonal(dist_matrix, np.inf)
    knn_scores: list[float] = []
    for i in range(N):
        sorted_dists = np.sort(dist_matrix[i])
        knn_scores.append(float(np.mean(sorted_dists[:actual_k])))

    return knn_scores


def find_knn_threshold_gmm(scores: np.ndarray) -> float:
    """
    在 log10(KNN_Dist) 上用 2-component GMM 找分界。

    低值群 = 比賽畫面（鏡位固定，鄰域緊密）
    高值群 = 非比賽畫面（視覺各異，鄰域稀疏）
    取兩群的機率交叉點為門檻，低於門檻的保留。
    """
    if scores.size == 0:
        return float("inf")
    if scores.size < 2:
        return float(np.max(scores))

    log_scores = np.log10(np.maximum(scores.astype(float), 1e-6))
    x = log_scores.reshape(-1, 1)

    try:
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=0,
            n_init=5,        # 多次初始化，避免陷入局部最優
        )
        gmm.fit(x)
    except ValueError:
        return float(np.max(scores))

    means = np.asarray(gmm.means_, dtype=float).ravel()
    variances = np.asarray(gmm.covariances_, dtype=float).ravel()
    weights = np.asarray(gmm.weights_, dtype=float).ravel()

    order = np.argsort(means)
    m1, m2 = means[order]
    v1, v2 = np.maximum(variances[order], 1e-12)
    w1, w2 = weights[order]

    s1, s2 = np.sqrt(v1), np.sqrt(v2)
    a = 1.0 / (2.0 * v1) - 1.0 / (2.0 * v2)
    b = m2 / v2 - m1 / v1
    c = (m1 ** 2) / (2.0 * v1) - (m2 ** 2) / (2.0 * v2) + np.log((s2 * w1) / (s1 * w2))

    roots = np.roots([a, b, c])
    real_roots = np.real(roots[np.isreal(roots)])
    between = real_roots[(real_roots > m1) & (real_roots < m2)]

    thresh_log = float(between[0]) if between.size > 0 else float((m1 + m2) / 2.0)
    return float(10.0 ** thresh_log)


def filter_segments_by_knn(
    segments: list[tuple[int, int]],
    knn_scores: list[float],
    threshold: float,
) -> list[tuple[int, int]]:
    """保留 KNN_Dist <= threshold 的片段（局部鄰域緊密 = 比賽畫面）。"""
    return [seg for seg, score in zip(segments, knn_scores) if score <= threshold]


# ---------------------------------------------------------------------------
# 輸出 CSV
# ---------------------------------------------------------------------------

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
            writer.writerow([
                idx,
                start_frame,
                end_frame,
                f"{start_sec:.3f}",
                f"{end_sec:.3f}",
                f"{end_sec - start_sec:.3f}",
            ])
            if idx % 20 == 0 or idx == seg_total:
                update_progress("輸出 CSV", idx, seg_total, start_time)


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    video_path = args.video_path or pick_default_video()
    output_csv = args.output_csv or f"{os.path.splitext(os.path.basename(video_path))[0]}_c4_segments.csv"

    print("=" * 60)
    print("bre6_c4: FrameDiff GMM + KNN 局部密度過濾")
    print("=" * 60)
    print(f"影片: {video_path}")
    print(f"輸出: {output_csv}")
    if args.diff_csv:
        print(f"FrameDiff CSV: {args.diff_csv}")

    # 步驟 1：計算 FrameDiff，GMM 找低動態分界
    if args.diff_csv:
        scores, times, fps, processed_frames = load_frame_diff_csv(args.diff_csv)
        print("已從 CSV 載入 frame diff，略過逐幀掃描")
    else:
        scores, times, fps, processed_frames = compute_frame_diff(video_path)
    frame_threshold = find_frame_threshold_gmm(scores)

    is_low = scores < frame_threshold
    if is_low.size > 0:
        is_low[0] = False  # 第一幀 diff=0，排除

    # 步驟 2：建構候選片段
    raw_segments = build_segments_from_mask(is_low)
    candidate_segments = merge_close_segments(raw_segments, MERGE_GAP_FRAMES)

    # 步驟 2.5：過濾過短片段（避免短片段進入 KNN）
    length_filtered_segments = filter_segments_by_min_duration(
        candidate_segments,
        fps=fps,
        min_duration_sec=MIN_SEGMENT_SEC,
    )

    # 步驟 3：擷取代表幀
    triplets, required_frames = collect_required_frames(length_filtered_segments)
    max_required = max(required_frames) if required_frames else 0
    frame_cache = load_required_gray_frames(video_path, required_frames, max_required)

    # 步驟 4：KNN 局部密度過濾
    knn_scores = compute_knn_density_scores(triplets, frame_cache, k=KNN_K)
    knn_threshold = find_knn_threshold_gmm(np.asarray(knn_scores, dtype=float))
    filtered_segments = filter_segments_by_knn(length_filtered_segments, knn_scores, knn_threshold)

    # 輸出
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
    print(f"長度過濾後(>= {MIN_SEGMENT_SEC:.1f}s): {len(length_filtered_segments)}")
    print(f"KNN_Dist GMM 分界值: {knn_threshold:.4f}")
    kept = sum(1 for s in knn_scores if s <= knn_threshold)
    print(f"KNN 保留/剔除: {kept} / {len(length_filtered_segments) - kept}")
    print(f"最終保留片段數: {len(filtered_segments)}")
    print("完成")


if __name__ == "__main__":
    main()