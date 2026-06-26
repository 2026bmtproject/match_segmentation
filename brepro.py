"""
brepro.py

功能摘要:
        讀取影片後，先計算逐幀 MAD(FrameDiff) 分數，再用 2-component GMM
        自動找出低動態門檻，擷取候選片段並進行跨片段代表幀比對，最後依
        Cross_Diff_Avg 門檻保留較穩定片段，輸出為精簡版 CSV。

輸入 / 輸出:
        - 輸入: 影片檔 (mp4)
        - 輸出: CSV 欄位僅包含
                Start_Frame, End_Frame, Start_Sec, End_Sec, Duration_Sec

使用方式:
        python brepro.py MK_vs_CT_2019.mp4 MK_vs_CT_2019.csv
        python brepro.py
                (若未給參數，會自動選目前目錄第一個 .mp4，
                 並輸出為 <影片名>_brepro_segments.csv)

        排除重跑模式:
        python brepro.py test2.mp4 r2.csv --exclude-csv r.csv
                (讀取上一輪輸出的 r.csv，略過其中所有 frame 後，
                 在剩餘 frame 中重新尋找穩定片段。適用於使用者
                 不滿意上一輪挑出的片段、想在其餘部分重找的情境。)

程式流程:
        1) 逐幀計算 FrameDiff(MAD)
        2) GMM 自動分界，取得低動態幀遮罩
        3) 連續低動態幀轉為片段，並合併近距離片段
        4) 過濾過短片段
        5) 每段取 2 個代表幀，做跨片段 MAD 比對
        6) 依 Cross_Diff_Avg 門檻篩選最終片段
        7) 輸出精簡欄位 CSV

可調整參數:
        - --merge-min-ratio (0~1, 預設 0.5)
            兩片段之間 gap 內低 diff 幀比例門檻。比例 >= 門檻時合併。

        - MIN_SEGMENT_SECONDS (預設 0.5)
            最短保留片段秒數。值越大，短片段會被更多地濾掉。

        - COMPARE_SIZE (預設 128x72)
            代表幀縮圖尺寸。較小可加速跨片段比對；較大可保留更多影像細節。

        - --avg-thr-scale (0~1, 預設 0.7)
            動態門檻放寬比例縮放因子。越小越嚴格、保留段數通常越少。

        - --avg-thr-pct (例如 0.05)
            直接指定固定放寬百分比，提供後會覆蓋動態比例策略。

        - --frame-step (預設 3)
            FrameDiff(MAD) 階段的跳幀取樣間隔。每 N 幀才實際解碼/比對一次，
            被跳過的幀沿用該區間算出的 MAD（區段邊界精度約 ±N 幀）。
            值越大越快，但邊界越粗；設 1 表示逐幀（不跳幀）。

進度顯示:
        使用 SmoothProgress 類別，採用 EWMA 平滑速率估計，使 ETA 更穩定。
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time

import cv2
import numpy as np


DEFAULT_MERGE_MIN_RATIO = 0.5
MIN_SEGMENT_SECONDS = 0.5
DEFAULT_AVG_THR_SCALE = 0.7
DEFAULT_FRAME_STEP = 3
COMPARE_SIZE = (128, 72)


class SmoothProgress:
    """終端進度條，使用平滑速率估計讓 ETA 更穩定。"""

    def __init__(
        self,
        label: str,
        total: int,
        width: int = 32,
        smoothing: float = 0.18,
        min_interval: float = 0.08,
    ) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.width = max(int(width), 10)
        self.smoothing = float(max(0.01, min(smoothing, 0.99)))
        self.min_interval = float(max(min_interval, 0.01))

        self.current = 0
        self.start_time = time.perf_counter()
        self.last_render_time = 0.0
        self.last_sample_time = self.start_time
        self.last_sample_value = 0
        self.ewma_rate = 0.0
        self.done = False

    def _build_bar(self, ratio: float) -> str:
        ratio = max(0.0, min(ratio, 1.0))
        filled = int(ratio * self.width)

        if filled >= self.width:
            return "=" * self.width
        if filled <= 0:
            return ">" + "." * (self.width - 1)
        return "=" * (filled - 1) + ">" + "." * (self.width - filled)

    def update(self, current: int, force: bool = False) -> None:
        now = time.perf_counter()
        current = max(self.current, min(int(current), self.total))

        dt = now - self.last_sample_time
        dn = current - self.last_sample_value
        if dt > 0 and dn > 0:
            inst_rate = dn / dt
            if self.ewma_rate <= 0.0:
                self.ewma_rate = inst_rate
            else:
                self.ewma_rate = self.smoothing * inst_rate + (1.0 - self.smoothing) * self.ewma_rate
            self.last_sample_time = now
            self.last_sample_value = current

        if not force and (now - self.last_render_time) < self.min_interval and current < self.total:
            self.current = current
            return

        ratio = current / self.total
        bar = self._build_bar(ratio)
        elapsed = max(now - self.start_time, 1e-9)
        rate = self.ewma_rate if self.ewma_rate > 0 else (current / elapsed)
        eta = 0.0 if current >= self.total else (self.total - current) / max(rate, 1e-9)

        print(
            f"\r{self.label:<18} [{bar}] {ratio * 100:6.2f}% "
            f"({current}/{self.total}) | {rate:7.2f}/s | ETA {eta:7.1f}s",
            end="",
            flush=True,
        )

        self.last_render_time = now
        self.current = current

        if current >= self.total and not self.done:
            self.done = True
            print()


def round_to_int(value: float) -> int:
    return int(np.rint(value))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def pick_default_video() -> str:
    videos = sorted(glob.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError("找不到 .mp4 檔案，請提供影片路徑")
    return videos[0]


def load_excluded_frames(csv_path: str) -> set[int]:
    """讀取先前輸出的片段 CSV，回傳所有被涵蓋的 frame 索引集合。

    用於「排除重跑」模式：使用者若不滿意上一輪挑出的片段，可把該 CSV
    當作排除清單，程式會略過這些 frame 後，在剩餘 frame 中重新尋找穩定片段。
    CSV 需含 Start_Frame / End_Frame 欄位（即本程式的標準輸出格式）。
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到排除用 CSV: {csv_path}")

    excluded: set[int] = set()
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "Start_Frame" not in reader.fieldnames or "End_Frame" not in reader.fieldnames:
            raise ValueError("排除用 CSV 缺少 Start_Frame / End_Frame 欄位")

        for row in reader:
            try:
                start_frame = int(float(row["Start_Frame"]))
                end_frame = int(float(row["End_Frame"]))
            except (TypeError, ValueError):
                continue

            if end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame
            excluded.update(range(start_frame, end_frame + 1))

    return excluded


def find_threshold_gmm(scores: np.ndarray) -> float:
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


def compute_frame_diff(
    video_path: str,
    frame_step: int = 1,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """逐幀計算 FrameDiff(MAD)，可用 frame_step 做跳幀取樣以加速。

    跳幀取樣: 每 frame_step 幀才實際解碼(retrieve)並比對一次 MAD，被跳過
    的幀以 cap.grab() 快速掠過(不做色彩轉換/複製)，並沿用該區間算出的 MAD
    分數回填。輸出陣列仍以「真實幀號」為索引，因此後續流程(片段、秒數、
    CSV、排除重跑)完全不需改動，僅區段邊界精度約 ±frame_step 幀。
    """
    step = max(int(frame_step), 1)

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

    bar = SmoothProgress("step 1: 掃描 FrameDiff(MAD)", total_frames)
    bar.update(1, force=True)

    frame_no = 0
    last_decode_idx = 0
    while True:
        grabbed = cap.grab()
        if not grabbed:
            break

        frame_no += 1
        diffs.append(0)  # 先佔位，待此區間解碼後回填
        times.append(frame_no / fps)

        if frame_no % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            score = round_to_int(float(np.mean(diff)))
            prev_gray = gray

            # 把本區間(上次解碼之後到這次解碼)所有被跳過的幀一併填上此分數
            for k in range(last_decode_idx + 1, frame_no + 1):
                diffs[k] = score
            last_decode_idx = frame_no

        bar.update(frame_no + 1)

    cap.release()

    # 結尾未滿一個 step 的殘餘幀，沿用最後一次算出的分數
    if last_decode_idx < frame_no:
        tail_score = diffs[last_decode_idx]
        for k in range(last_decode_idx + 1, frame_no + 1):
            diffs[k] = tail_score

    processed = frame_no + 1
    bar.update(processed, force=True)

    return np.array(diffs, dtype=float), np.array(times, dtype=float), float(fps), processed


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


def merge_close_segments(
    segments: list[tuple[int, int]],
    is_low: np.ndarray,
    min_ratio: float,
) -> list[tuple[int, int]]:
    if not segments:
        return []

    min_ratio = clamp(min_ratio, 0.0, 1.0)
    merged = [segments[0]]

    for start_frame, end_frame in segments[1:]:
        prev_start, prev_end = merged[-1]
        gap = start_frame - prev_end - 1

        if gap <= 0:
            merged[-1] = (prev_start, max(prev_end, end_frame))
            continue

        gap_start = prev_end + 1
        gap_end_exclusive = start_frame
        gap_mask = is_low[gap_start:gap_end_exclusive]
        low_count = int(np.sum(gap_mask))
        low_ratio = low_count / max(gap, 1)

        if low_ratio >= min_ratio:
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
    bar = SmoothProgress("step 2: 擷取代表幀", total_to_scan)
    bar.update(0, force=True)

    frame_cache: dict[int, np.ndarray] = {}
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
        bar.update(min(frame_idx, total_to_scan))

    cap.release()
    bar.update(min(frame_idx, total_to_scan), force=True)

    missing = required_frames.difference(frame_cache.keys())
    if missing:
        raise RuntimeError(f"代表幀讀取失敗，缺少 {len(missing)} 幀")

    return frame_cache


def build_segment_vectors(
    pairs: list[tuple[int, int]],
    frame_cache: dict[int, np.ndarray],
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []

    for m1, m2 in pairs:
        f1 = cv2.resize(frame_cache[m1], COMPARE_SIZE, interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(frame_cache[m2], COMPARE_SIZE, interpolation=cv2.INTER_AREA)

        seg_vec = np.stack([f1.reshape(-1), f2.reshape(-1)]).astype(np.int16)
        vectors.append(seg_vec)

    return vectors


def compute_cross_segment_scores(
    pairs: list[tuple[int, int]],
    frame_cache: dict[int, np.ndarray],
) -> tuple[list[int], list[int], int]:
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
    bar = SmoothProgress("step 3: 跨片段比對(MAD)", total_pairs)
    bar.update(0, force=True)

    for i in range(seg_count):
        vi = vectors[i]
        for j in range(i + 1, seg_count):
            vj = vectors[j]
            mad_matrix = np.mean(np.abs(vi[:, None, :] - vj[None, :, :]), axis=2)
            pair_sum = float(np.sum(mad_matrix))
            sums[i] += pair_sum
            sums[j] += pair_sum

            done_pairs += 1
            bar.update(done_pairs)

    bar.update(total_pairs, force=True)

    avgs = sums / (compared_segments * 4)
    sums_int = [round_to_int(x) for x in sums.tolist()]
    avgs_int = [round_to_int(x) for x in avgs.tolist()]
    return sums_int, avgs_int, compared_segments


def suggest_avg_threshold_pct(segment_count: int) -> float:
    n = max(segment_count, 0)
    if n <= 50:
        return 0.01 + 0.04 * (n / 50.0)
    if n <= 100:
        return 0.05 + 0.03 * ((n - 50) / 50.0)
    return min(0.08 + 0.02 * ((n - 100) / 100.0), 0.10)


def compute_avg_threshold(
    cross_avgs: list[int],
    segment_count: int,
    scale: float,
    fixed_pct: float | None,
) -> tuple[int, int, float]:
    if not cross_avgs:
        return 0, 0, 0.0

    min_avg = int(min(cross_avgs))
    if fixed_pct is not None:
        pct = clamp(fixed_pct, 0.0, 0.20)
    else:
        pct = suggest_avg_threshold_pct(segment_count) * clamp(scale, 0.0, 1.0)

    threshold = round_to_int(min_avg * (1.0 + pct))
    return min_avg, threshold, pct


def filter_segments_by_cross_avg(
    segments: list[tuple[int, int]],
    pairs: list[tuple[int, int]],
    cross_sums: list[int],
    cross_avgs: list[int],
    threshold: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    keep_indices = [idx for idx, avg in enumerate(cross_avgs) if avg <= threshold]

    filtered_segments = [segments[idx] for idx in keep_indices]
    filtered_pairs = [pairs[idx] for idx in keep_indices]
    filtered_cross_sums = [cross_sums[idx] for idx in keep_indices]
    filtered_cross_avgs = [cross_avgs[idx] for idx in keep_indices]

    return filtered_segments, filtered_pairs, filtered_cross_sums, filtered_cross_avgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FrameDiff(MAD) + GMM + 每段2幀跨片段比對篩選",
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
        help="輸出 CSV 路徑（預設 <影片名>_brepro_segments.csv）",
    )
    parser.add_argument(
        "--avg-thr-scale",
        dest="avg_thr_scale",
        type=float,
        default=DEFAULT_AVG_THR_SCALE,
        help="Cross_Diff_Avg 門檻百分比強度(0~1，越小越嚴格)",
    )
    parser.add_argument(
        "--merge-min-ratio",
        dest="merge_min_ratio",
        type=float,
        default=DEFAULT_MERGE_MIN_RATIO,
        help="片段 gap 內低 diff 幀比例門檻(0~1，比例 >= 門檻則合併)",
    )
    parser.add_argument(
        "--avg-thr-pct",
        dest="avg_thr_pct",
        type=float,
        default=None,
        help="直接指定門檻百分比（例如 0.05 代表 +5%%）",
    )
    parser.add_argument(
        "--exclude-csv",
        dest="exclude_csv",
        default=None,
        help="排除重跑模式：指定先前輸出的片段 CSV，程式會略過其中所有 frame 後在剩餘 frame 重新尋找片段",
    )
    parser.add_argument(
        "--frame-step",
        dest="frame_step",
        type=int,
        default=DEFAULT_FRAME_STEP,
        help="FrameDiff(MAD) 跳幀取樣間隔(預設 3；每 N 幀解碼比對一次，設 1 表示逐幀)",
    )
    return parser.parse_args()


def write_segments_csv(
    output_csv: str,
    segments: list[tuple[int, int]],
    fps: float,
) -> None:
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Start_Frame",
            "End_Frame",
            "Start_Sec",
            "End_Sec",
            "Duration_Sec",
        ])

        total = max(len(segments), 1)
        bar = SmoothProgress("輸出 CSV", total)
        bar.update(0, force=True)

        for idx, (start_frame, end_frame) in enumerate(segments, start=1):
            start_sec = start_frame / fps
            end_sec = end_frame / fps
            duration_sec = end_sec - start_sec

            writer.writerow([
                start_frame,
                end_frame,
                f"{start_sec:.3f}",
                f"{end_sec:.3f}",
                f"{duration_sec:.3f}",
            ])

            bar.update(idx)

        bar.update(total, force=True)


def main() -> None:
    args = parse_args()

    video_path = args.video_path if args.video_path else pick_default_video()
    output_csv = (
        args.output_csv
        if args.output_csv
        else f"{os.path.splitext(os.path.basename(video_path))[0]}_brepro_segments.csv"
    )

    print("=" * 72)
    print("brepro: FrameDiff(MAD) + GMM + 跨片段比對篩選")
    print("=" * 72)
    print(f"影片:  {video_path}")
    print(f"輸出: {output_csv}")
    frame_step = max(int(args.frame_step), 1)
    if frame_step > 1:
        print(f"跳幀取樣: 每 {frame_step} 幀解碼比對一次")
    else:
        print("跳幀取樣: 逐幀（不跳幀）")

    excluded_frames: set[int] = set()
    if args.exclude_csv:
        excluded_frames = load_excluded_frames(args.exclude_csv)
        print(f"排除重跑: {args.exclude_csv}（排除 {len(excluded_frames)} 個 frame）")

    scores, times, fps, processed_frames = compute_frame_diff(video_path, frame_step)

    excluded_mask = np.zeros(scores.size, dtype=bool)
    if excluded_frames:
        valid_excluded = [f for f in excluded_frames if 0 <= f < scores.size]
        excluded_mask[valid_excluded] = True

    # 僅用未被排除的 frame 計算 GMM 門檻，讓重跑真正只在剩餘 frame 中尋找穩定片段
    threshold = find_threshold_gmm(scores[~excluded_mask])
    is_low = scores < threshold
    if is_low.size > 0:
        is_low[0] = False
    # 把被排除的 frame 標為非低動態，使其無法形成片段、也無法作為合併的橋接
    is_low[excluded_mask] = False

    raw_segments = build_segments_from_mask(is_low)
    merged_segments = merge_close_segments(raw_segments, is_low, args.merge_min_ratio)
    candidate_segments = filter_short_segments(merged_segments, fps, MIN_SEGMENT_SECONDS)

    pairs, required_frames = collect_required_frames(candidate_segments)
    max_required = max(required_frames) if required_frames else 0
    frame_cache = load_required_gray_frames(video_path, required_frames, max_required)

    cross_sums, cross_avgs, compared_segments = compute_cross_segment_scores(pairs, frame_cache)

    min_avg, avg_threshold, used_pct = compute_avg_threshold(
        cross_avgs,
        len(candidate_segments),
        args.avg_thr_scale,
        args.avg_thr_pct,
    )
    filtered_segments, _, _, _ = filter_segments_by_cross_avg(
        candidate_segments,
        pairs,
        cross_sums,
        cross_avgs,
        avg_threshold,
    )

    write_segments_csv(output_csv, filtered_segments, fps)

    low_frames = int(np.sum(is_low))
    duration_sec = float(times[-1]) if times.size else 0.0

    print("-" * 72)
    print(f"FPS: {fps:.3f}")
    print(f"總幀數: {processed_frames}")
    print(f"總時長: {duration_sec / 60:.1f} 分鐘")
    print(f"GMM 分界值: {threshold:.2f}")
    if excluded_frames:
        print(f"排除重跑略過 frame 數: {int(np.sum(excluded_mask))}")
    print(f"低於分界: {low_frames} ({low_frames / max(scores.size, 1) * 100:.1f}%)")
    print(f"原始片段數: {len(raw_segments)}")
    print(f"合併後片段數: {len(merged_segments)}")
    print(f"合併門檻(gap低diff比例): {clamp(args.merge_min_ratio, 0.0, 1.0):.2f}")
    print(f"候選片段數(最短 {MIN_SEGMENT_SECONDS:.1f} 秒): {len(candidate_segments)}")
    if cross_avgs:
        print(f"Cross_Diff_Avg 最小值: {min_avg}")
        print(f"Cross_Diff_Avg 篩選百分比: {used_pct * 100:.2f}%")
        print(f"Cross_Diff_Avg 篩選門檻: {avg_threshold}")
    else:
        print("Cross_Diff_Avg 篩選: 無候選片段可計算")
    print(f"參照片段數: {compared_segments}")
    print(f"最終保留片段數: {len(filtered_segments)}")
    print(f"代表幀快取數: {len(frame_cache)}")
    print("完成")


if __name__ == "__main__":
    main()
