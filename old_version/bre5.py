"""
precise_rally_detector.py  ─  精準固定視角片段偵測器

演算法三階段：
  Stage 1 │ GMM 自動閾值 + 原始幀差取初始候選片段（無合併、無寬容）
  Stage 2 │ 跨片段靜止幀 cross-diff 驗證：相同視角的靜止幀互比差值極低
           │ → 以多數票（主視角聚類）淘汰異視角片段（回放、觀眾席等）
  Stage 3 │ 精確邊界修剪：從原始（未平滑）幀差往內縮，確保首尾無高差值幀

輸出 CSV 欄位：
  Segment_ID, Start_Frame, End_Frame, Start_Sec, End_Sec, Duration_Sec

用法：
    # 方式 A：直接傳影片（程式自動計算幀差）
    python precise_rally_detector.py input.mp4

    # 方式 B：傳影片 + 預先算好的幀差 CSV（跳過幀差計算，較快）
    python precise_rally_detector.py input.mp4 diff.csv

    # 方式 C：完整指定輸出路徑
    python precise_rally_detector.py input.mp4 diff.csv segments.csv
"""

import cv2
import csv
import sys
import os
import glob
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# ─────────────────────────────────────────────────────────────
#  可調整參數
# ─────────────────────────────────────────────────────────────

# 最短有效片段（秒）：短於此長度視為雜訊，直接丟棄
MIN_SEGMENT_SEC = 2.0

# 最短有效片段（秒）用於第一次粗篩（可稍寬鬆）
MIN_SEGMENT_SEC_ROUGH = 0.5

# 精確邊界修剪：從邊界向內縮時，允許的連續「雜訊高差值幀」數量
# 設為 0 = 完全精準，設為 2~3 = 允許邊緣少量過渡幀（更穩健）
BOUNDARY_NOISE_TOLERANCE = 0

# 跨片段驗證：縮放比例（加速比較，不影響準確度）
CROSS_DIFF_RESIZE = (160, 90)

# 跨片段驗證：選每個片段幾個「靜止幀」做代表
STATIC_FRAMES_PER_SEGMENT = 3

# 跨片段驗證：cross-diff 閾值（同視角預期極低）
# None = 自動以所有跨片段 diff 的分佈找分界
CROSS_DIFF_THRESHOLD = None

# 跨片段驗證：最多取幾個片段做 pairwise 比較（避免片段數量多時過慢）
MAX_SEGMENTS_FOR_CROSS = 100

# 邊界修剪：允許邊緣幾幀偶發雜訊（0=最嚴格）
BOUNDARY_NOISE_TOLERANCE = 1

# ─────────────────────────────────────────────────────────────


def compute_diff_scores(video_path: str) -> tuple[np.ndarray, float]:
    """直接從影片計算逐幀差值，回傳 (scores array, fps)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[幀差計算] {video_path}  FPS={fps:.2f}  總幀數={total:,}")

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("無法讀取第一幀")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    scores = [0.0]
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        scores.append(float(np.sum(diff)))
        prev_gray = gray
        if frame_idx % 300 == 0:
            print(f"  已處理 {frame_idx}/{total} 幀...", end="\r")

    cap.release()
    print(f"\n[幀差計算] 完成，共 {frame_idx} 幀")
    return np.array(scores), fps


def load_diff_csv(csv_path: str) -> tuple[np.ndarray, float]:
    """從 analyze_diff.py 產生的 CSV 讀取幀差資料，回傳 (scores, fps)"""
    df = pd.read_csv(csv_path)
    scores = df["Difference_Score"].values.astype(float)
    times = df["Time_Sec"].values.astype(float)
    diffs = np.diff(times[times > 0])
    fps = 1.0 / np.median(diffs[diffs > 0]) if len(diffs) > 0 else 30.0
    return scores, fps


def find_threshold_gmm(scores: np.ndarray) -> float:
    """GMM 自動找 rally / non-rally 分界（log scale）"""
    log_scores = np.log10(np.maximum(scores, 1)).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(log_scores)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_

    order = np.argsort(means)
    m1, m2 = means[order]
    v1, v2 = variances[order]
    w1, w2 = weights[order]
    s1, s2 = np.sqrt(v1), np.sqrt(v2)

    a = 1 / (2 * v1) - 1 / (2 * v2)
    b = m2 / v2 - m1 / v1
    c = m1**2 / (2 * v1) - m2**2 / (2 * v2) + np.log((s2 * w1) / (s1 * w2))

    roots = np.roots([a, b, c])
    valid = roots[(np.real(roots) > m1) & (np.real(roots) < m2)]
    thresh_log = float(np.real(valid[0])) if len(valid) > 0 else (m1 + m2) / 2
    threshold = 10**thresh_log

    print(f"[GMM閾值] low_mean=10^{m1:.2f}={10**m1:.0f}  "
          f"high_mean=10^{m2:.2f}={10**m2:.0f}  "
          f"threshold={threshold:.0f}")
    return threshold


# ─────────────────────────────────────────────────────────────
#  Stage 1：初始粗分段（嚴格、無合併）
# ─────────────────────────────────────────────────────────────

def rough_segments(scores: np.ndarray, fps: float, threshold: float) -> list[tuple[int, int]]:
    """
    直接用原始幀差找「連續低差值段落」，不做任何合併。
    只過濾極短片段（MIN_SEGMENT_SEC_ROUGH）。
    """
    min_len = max(1, int(fps * MIN_SEGMENT_SEC_ROUGH))
    in_low = scores < threshold

    segs = []
    start = None
    for i, flag in enumerate(in_low):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(scores) - 1))

    segs = [(s, e) for s, e in segs if (e - s + 1) >= min_len]
    print(f"[Stage 1] 粗分段：找到 {len(segs)} 個候選片段（threshold={threshold:.0f}）")
    return segs


# ─────────────────────────────────────────────────────────────
#  Stage 2：跨片段 cross-diff 驗證
#
#  效能關鍵：把所有需要讀取的幀號收集好、排序後「一次順序讀完」存進記憶體，
#  再用純 numpy 矩陣運算做所有 pairwise 比較。
#  seek 次數從 O(n_pairs × n_static²) 降到 O(n_frames_needed)。
# ─────────────────────────────────────────────────────────────

def _progress_bar(current: int, total: int, width: int = 40, prefix: str = "") -> None:
    """簡易進度條，不依賴 tqdm。"""
    filled = int(width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * current / total if total > 0 else 0
    print(f"\r  {prefix}[{bar}] {current}/{total}  {pct:.1f}%", end="", flush=True)


def get_static_frames(scores: np.ndarray, start: int, end: int, n: int) -> list[int]:
    """
    從片段 [start, end] 找 n 個「最靜止」代表幀（幀差最低的幀）。
    分成 n 個等分區間，各取最低差值幀，避免全選在同一區域。
    """
    seg_len = end - start + 1
    if seg_len <= n:
        return list(range(start, end + 1))

    frames = []
    chunk = seg_len // n
    for k in range(n):
        lo = start + k * chunk
        hi = start + (k + 1) * chunk if k < n - 1 else end + 1
        sub = scores[lo:hi]
        best = lo + int(np.argmin(sub))
        frames.append(best)
    return frames


def _load_frames_sequential(
    video_path: str,
    needed_frames: list[int],
    size: tuple[int, int],
) -> dict[int, np.ndarray]:
    """
    以順序讀取方式一次把所有需要的幀讀入記憶體。

    needed_frames 必須已排序（由小到大）。
    回傳 {frame_no: flat_float32_array}。
    """
    frame_set = set(needed_frames)
    frame_dict: dict[int, np.ndarray] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frame_dict

    total_needed = len(needed_frames)
    loaded = 0
    cur_frame = 0

    print(f"\n  [載入幀] 需讀取 {total_needed} 幀（順序掃描，無 seek）")
    _progress_bar(0, total_needed, prefix="載入 ")

    # 用 grab()+retrieve() 跳過不需要的幀（比 read() 快，不解碼）
    max_frame = needed_frames[-1] if needed_frames else 0

    while cur_frame <= max_frame:
        if cur_frame in frame_set:
            ret, img = cap.read()
            if ret:
                small = cv2.resize(img, size).astype(np.float32)
                frame_dict[cur_frame] = small.reshape(-1)   # 展平為 1D 向量
            loaded += 1
            _progress_bar(loaded, total_needed, prefix="載入 ")
        else:
            cap.grab()   # 只讀 header，不解碼，速度快很多
        cur_frame += 1

    cap.release()
    print()   # 換行
    return frame_dict


def validate_by_cross_diff(
    video_path: str,
    segments: list[tuple[int, int]],
    scores: np.ndarray,
    fps_hint: float = 30.0,
) -> list[tuple[int, int]]:
    """
    跨片段靜止幀比對驗證。

    邏輯：
    - 同一固定視角的片段，其「靜止幀」互比差值應很低（背景相同）
    - 回放、廣告、觀眾席等鏡頭，與主視角靜止幀比對差值會很高
    - 以所有 pairwise cross-diff 分佈，自動找閾值分出「主視角群」
    - 多數票：與多數其他片段 cross-diff 低的 → 主視角，保留

    效能優化：
    - 一次順序讀完所有需要的幀（避免重複 seek）
    - numpy 廣播矩陣運算一次計算所有 pairwise L1 距離
    """
    if len(segments) <= 1:
        return segments

    n_seg = len(segments)
    print(f"[Stage 2] 跨片段驗證：共 {n_seg} 個片段")

    # 若片段過多，採樣避免過慢（優先選較長的片段）
    if n_seg > MAX_SEGMENTS_FOR_CROSS:
        lengths = [e - s for s, e in segments]
        idx_sorted = np.argsort(lengths)[::-1][:MAX_SEGMENTS_FOR_CROSS]
        working_segs = [segments[i] for i in sorted(idx_sorted)]
        working_idx = sorted(idx_sorted)
        print(f"  → 片段過多，採樣最長 {MAX_SEGMENTS_FOR_CROSS} 個做驗證")
    else:
        working_segs = segments
        working_idx = list(range(n_seg))

    n_work = len(working_segs)

    # ── 1. 找每個片段的靜止幀號 ──────────────────────────────
    static_frames_per_seg: list[list[int]] = []
    for s, e in working_segs:
        sf = get_static_frames(scores, s, e, STATIC_FRAMES_PER_SEGMENT)
        static_frames_per_seg.append(sf)

    # 所有需要讀取的幀號（排序，準備順序掃描）
    all_needed: list[int] = sorted({f for sf in static_frames_per_seg for f in sf})

    # ── 2. 一次順序讀完所有幀 ────────────────────────────────
    frame_dict = _load_frames_sequential(video_path, all_needed, CROSS_DIFF_RESIZE)

    if not frame_dict:
        print("  [警告] 無法讀取任何幀，跳過驗證")
        return segments

    # ── 3. 每個片段的靜止幀向量清單 ─────────────────────────────
    #
    # 關鍵：保留每個片段所有靜止幀向量（不壓縮成單幀）。
    # cross_matrix[i,j] = MIN over 所有 (a∈frames_i, b∈frames_j) 的 mean_L1_diff
    # 這讓我們找到兩個片段「最像的那一刻」，對應到球員都不在前景的瞬間。
    # 同視角片段的最佳配對差值極低；不同視角再怎麼配都差值很高。
    #
    D = CROSS_DIFF_RESIZE[0] * CROSS_DIFF_RESIZE[1] * 3   # 每幀向量維度

    seg_vecs: list[np.ndarray] = []   # 每個元素 shape (k_i, D)
    for sf_list in static_frames_per_seg:
        available = [frame_dict[f] for f in sf_list if f in frame_dict]
        if available:
            seg_vecs.append(np.stack(available))        # (k, D)
        else:
            seg_vecs.append(np.zeros((1, D), dtype=np.float32))   # fallback

    print(f"\n  [矩陣計算] 計算 {n_work}×{n_work} min-cross-diff 矩陣"
          f"（每片段 {STATIC_FRAMES_PER_SEGMENT} 幀）...")

    cross_matrix = np.full((n_work, n_work), np.inf, dtype=np.float32)
    np.fill_diagonal(cross_matrix, 0.0)

    _progress_bar(0, n_work, prefix="cross-diff ")
    for i in range(n_work):
        Vi = seg_vecs[i]                        # (ki, D)
        for j in range(i + 1, n_work):
            Vj = seg_vecs[j]                    # (kj, D)
            # diff_mat[a,b] = mean_L1(Vi[a], Vj[b])
            # shape: (ki, kj)  —— 廣播：(ki,1,D) - (1,kj,D)
            diff_mat = np.mean(
                np.abs(Vi[:, None, :] - Vj[None, :, :]),
                axis=2
            )
            min_d = float(diff_mat.min())
            cross_matrix[i, j] = min_d
            cross_matrix[j, i] = min_d
        _progress_bar(i + 1, n_work, prefix="cross-diff ")
    print()

    # ── 4. 自動閾值（GMM，帶自動降級 fallback）───────────────
    all_diffs = cross_matrix[np.triu_indices(n_work, k=1)]
    all_diffs = all_diffs[np.isfinite(all_diffs) & (all_diffs > 0)]

    p10, p25, p50, p75, p90 = np.percentile(all_diffs, [10, 25, 50, 75, 90])
    print(f"  [cross-diff 分佈] p10={p10:.1f}  p25={p25:.1f}  p50={p50:.1f}"
          f"  p75={p75:.1f}  p90={p90:.1f}")

    if CROSS_DIFF_THRESHOLD is not None:
        cd_thresh = float(CROSS_DIFF_THRESHOLD)
        print(f"  [cross-diff 閾值] 手動: {cd_thresh:.2f}")
    else:
        cd_thresh = None
        if len(all_diffs) >= 8:
            try:
                log_d = np.log1p(all_diffs).reshape(-1, 1)
                gmm2 = GaussianMixture(n_components=2, covariance_type="full",
                                       random_state=0, n_init=3)
                gmm2.fit(log_d)
                means2 = gmm2.means_.flatten()
                order2 = np.argsort(means2)
                m_lo, m_hi = means2[order2]
                gap = m_hi - m_lo
                # 若兩峰距離太近（< 0.3 log units），GMM 分不開，退回百分位
                if gap >= 0.3:
                    cd_thresh = float(np.expm1(m_lo + gap * 0.6))
                    print(f"  [cross-diff 閾值] GMM: {cd_thresh:.2f}  "
                          f"(low={np.expm1(m_lo):.1f}  high={np.expm1(m_hi):.1f}"
                          f"  gap={gap:.2f})")
                else:
                    print(f"  [cross-diff 閾值] GMM 分佈重疊（gap={gap:.2f}），"
                          f"退回百分位策略")
            except Exception as e:
                print(f"  [cross-diff 閾值] GMM 失敗（{e}），退回百分位策略")

        if cd_thresh is None:
            # 百分位策略：取 p25~p50 之間，偏向保守（寧可多留）
            cd_thresh = float(p50)
            print(f"  [cross-diff 閾值] 百分位 p50: {cd_thresh:.2f}")

    # ── 5. 多數票投票 ─────────────────────────────────────────
    # 票數 = 與幾個其他片段的 min-cross-diff ≤ cd_thresh
    similar_votes = np.sum(cross_matrix <= cd_thresh, axis=1) - 1   # 去掉自身

    # 動態多數門檻：至少要有 1/4 的其他片段認同
    majority = max(1, n_work // 4)
    is_main_view = similar_votes >= majority

    print(f"\n  [投票] 門檻 = {majority} 票（共 {n_work} 個片段）")
    print(f"  {'#':>4}  {'起始幀':>8}  {'結束幀':>8}  {'時長(s)':>8}  {'票數':>6}  {'狀態'}")
    print(f"  {'-'*55}")
    for i, (s, e) in enumerate(working_segs):
        dur = (e - s + 1) / max(fps_hint, 1)
        status = "✓ 保留" if is_main_view[i] else "✗ 排除"
        print(f"  {i+1:>4}  {s:>8d}  {e:>8d}  {dur:>8.1f}  {similar_votes[i]:>6d}  {status}")

    # ── 6. 安全保護：若保留比例過低，自動放寬閾值重試 ──────────
    keep_ratio = is_main_view.sum() / n_work
    if keep_ratio < 0.25:
        print(f"\n  [警告] 保留比例 {keep_ratio:.0%} 過低，以 p75 閾值重試...")
        cd_thresh_relaxed = float(p75)
        similar_votes = np.sum(cross_matrix <= cd_thresh_relaxed, axis=1) - 1
        majority_relaxed = max(1, n_work // 5)
        is_main_view = similar_votes >= majority_relaxed
        print(f"  [重試] 閾值={cd_thresh_relaxed:.2f}  門檻={majority_relaxed} 票"
              f"  → 保留 {is_main_view.sum()}/{n_work}")

    # ── 7. 回傳已驗證片段 ────────────────────────────────────
    if n_seg > MAX_SEGMENTS_FOR_CROSS:
        validated_working = {working_idx[i] for i in range(n_work) if is_main_view[i]}
        unvalidated = set(range(n_seg)) - set(working_idx)
        result = [segments[i] for i in range(n_seg)
                  if i in validated_working or i in unvalidated]
    else:
        result = [segments[i] for i in range(n_seg) if is_main_view[i]]

    print(f"\n[Stage 2] 驗證後保留 {len(result)}/{n_seg} 個片段")
    return result


# ─────────────────────────────────────────────────────────────
#  Stage 3：精確邊界修剪
# ─────────────────────────────────────────────────────────────

def trim_boundary(scores: np.ndarray, start: int, end: int,
                  threshold: float) -> tuple[int, int]:
    """
    從片段 [start, end] 的兩端向內找到第一個符合 score < threshold 的幀，
    允許 BOUNDARY_NOISE_TOLERANCE 幀的偶發雜訊（連續高差值幀）。

    Stage 1 已確保片段內大部分幀 < threshold，所以通常只修剪極少幀。
    """
    tol = BOUNDARY_NOISE_TOLERANCE
    n = end - start + 1

    # ── 左端：從 start 往右找，連續 tol+1 幀皆 < threshold 才算穩定起點
    new_start = start
    found = False
    consecutive_low = 0
    for i in range(start, end + 1):
        if scores[i] < threshold:
            consecutive_low += 1
            if consecutive_low >= tol + 1:
                new_start = i - tol   # 回退到連續段的起點
                found = True
                break
        else:
            consecutive_low = 0
    if not found:
        return start, end   # 整段都沒有穩定低差值區域，保留原始

    # ── 右端：從 end 往左找，同上
    new_end = end
    consecutive_low = 0
    for i in range(end, new_start - 1, -1):
        if scores[i] < threshold:
            consecutive_low += 1
            if consecutive_low >= tol + 1:
                new_end = i + tol   # 回退到連續段的末端
                break
        else:
            consecutive_low = 0

    new_start = max(start, new_start)
    new_end = min(end, new_end)

    if new_start > new_end:
        return start, end
    return new_start, new_end


def precise_trim_all(segments: list[tuple[int, int]],
                     scores: np.ndarray,
                     fps: float,
                     threshold: float) -> list[tuple[int, int]]:
    """對所有片段進行精確邊界修剪，並再次過濾過短片段。"""
    min_len = max(1, int(fps * MIN_SEGMENT_SEC))
    trimmed = []
    for s, e in segments:
        ns, ne = trim_boundary(scores, s, e, threshold)
        if (ne - ns + 1) >= min_len:
            trimmed.append((ns, ne))

    print(f"[Stage 3] 精確修剪後：{len(trimmed)} 個片段（最短 {MIN_SEGMENT_SEC}s）")
    return trimmed


# ─────────────────────────────────────────────────────────────
#  輸出 CSV
# ─────────────────────────────────────────────────────────────

def write_segments_csv(segments: list[tuple[int, int]],
                       fps: float, output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Segment_ID", "Start_Frame", "End_Frame",
            "Start_Sec", "End_Sec", "Duration_Sec"
        ])
        for i, (s, e) in enumerate(segments, 1):
            start_sec = round(s / fps, 3)
            end_sec = round(e / fps, 3)
            duration = round((e - s + 1) / fps, 3)
            writer.writerow([i, s, e, start_sec, end_sec, duration])
    print(f"\n[輸出] 片段 CSV 已儲存 → {output_path}")


def print_summary(segments: list[tuple[int, int]], fps: float,
                  total_frames: int) -> None:
    total_dur = sum((e - s + 1) / fps for s, e in segments)
    total_src = total_frames / fps
    print("\n" + "=" * 60)
    print(f"  共 {len(segments)} 個固定視角片段  "
          f"（{total_dur:.1f}s / 原片 {total_src:.1f}s = "
          f"{total_dur/total_src*100:.1f}%）")
    print("=" * 60)
    print(f"  {'#':>4}  {'起始幀':>8}  {'結束幀':>8}  "
          f"{'起始時間':>10}  {'結束時間':>10}  {'時長(s)':>8}")
    print(f"  {'-'*58}")
    for i, (s, e) in enumerate(segments, 1):
        ss = s / fps
        es = e / fps
        dur = (e - s + 1) / fps
        sm, ss_ = divmod(ss, 60)
        em, es_ = divmod(es, 60)
        print(f"  {i:>4}  {s:>8d}  {e:>8d}  "
              f"  {int(sm):02d}:{ss_:05.2f}    {int(em):02d}:{es_:05.2f}  {dur:>8.2f}")


# ─────────────────────────────────────────────────────────────
#  主程式
# ─────────────────────────────────────────────────────────────

def main():
    # ── 參數處理 ──────────────────────────────────────────────
    video_path  = sys.argv[1] if len(sys.argv) > 1 else None
    diff_csv    = sys.argv[2] if len(sys.argv) > 2 else None
    output_csv  = sys.argv[3] if len(sys.argv) > 3 else None

    if not video_path:
        mp4s = sorted(glob.glob("*.mp4"))
        if not mp4s:
            print("[錯誤] 找不到 .mp4 檔案，請指定路徑")
            sys.exit(1)
        video_path = mp4s[0]
        print(f"[自動] 使用影片: {video_path}")

    if not diff_csv:
        # 只嘗試同影片名稱的 CSV，避免誤用其他影片的 *diff*.csv
        base = os.path.splitext(video_path)[0]
        candidates = [f"{base}.csv", f"{base}_diff.csv"]
        for c in candidates:
            if os.path.exists(c):
                diff_csv = c
                print(f"[自動] 使用幀差 CSV: {diff_csv}")
                break

    if not output_csv:
        base = os.path.splitext(video_path)[0]
        output_csv = f"{base}_segments.csv"

    print("=" * 60)
    print("  精準固定視角片段偵測器")
    print("=" * 60)
    print(f"  影片: {video_path}")
    print(f"  幀差: {diff_csv or '（將自動計算）'}")
    print(f"  輸出: {output_csv}")
    print("-" * 60)

    # ── 取得幀差資料 ──────────────────────────────────────────
    if diff_csv and os.path.exists(diff_csv):
        scores, fps = load_diff_csv(diff_csv)
        print(f"[載入] FPS={fps:.2f}  總幀數={len(scores):,}")
    else:
        scores, fps = compute_diff_scores(video_path)
        # 順手儲存幀差 CSV 供下次重用
        base = os.path.splitext(video_path)[0]
        auto_csv = f"{base}_diff.csv"
        times = np.arange(len(scores)) / fps
        df_save = pd.DataFrame({
            "Frame": np.arange(len(scores)),
            "Time_Sec": np.round(times, 3),
            "Difference_Score": scores.astype(int)
        })
        df_save.to_csv(auto_csv, index=False)
        print(f"[儲存] 幀差 CSV 已存至 {auto_csv}（供下次重用）")

    total_frames = len(scores)

    # ── Stage 1：GMM 閾值 + 初始粗分段 ───────────────────────
    threshold = find_threshold_gmm(scores)
    segments = rough_segments(scores, fps, threshold)

    if not segments:
        print("[警告] 未找到任何候選片段，請檢查影片或降低閾值。")
        sys.exit(0)

    # ── Stage 2：跨片段 cross-diff 驗證 ──────────────────────
    segments = validate_by_cross_diff(video_path, segments, scores, fps_hint=fps)

    if not segments:
        print("[警告] 跨片段驗證後無保留片段。")
        sys.exit(0)

    # ── Stage 3：精確邊界修剪 ─────────────────────────────────
    segments = precise_trim_all(segments, scores, fps, threshold)

    if not segments:
        print("[警告] 精確修剪後無保留片段。")
        sys.exit(0)

    # ── 輸出 ──────────────────────────────────────────────────
    print_summary(segments, fps, total_frames)
    write_segments_csv(segments, fps, output_csv)


if __name__ == "__main__":
    main()