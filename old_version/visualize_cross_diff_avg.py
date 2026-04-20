"""
visualize_cross_diff_avg.py

視覺化 cross_diff_avg 計算流程：
1) 先用 bre6_c3 同邏輯找候選片段（FrameDiff + 2-component GMM + merge）
2) 為每個候選片段取 1 個代表幀
3) 在同一個畫面同時顯示：
   - 多個候選片段的代表幀
   - 片段彼此的 pairwise frame diff 矩陣
   - 每個片段的 cross_diff_avg（在目前顯示片段子集合內）

用法：
    python visualize_cross_diff_avg.py test2.mp4
    python visualize_cross_diff_avg.py test2.mp4 --max-segments 9 --output cross_diff_avg.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from sklearn.mixture import GaussianMixture


MERGE_GAP_FRAMES = 3
BAR_WIDTH = 30
COMPARE_SIZE = (128, 72)


def configure_plot_font() -> None:
    # 針對 Windows 環境優先使用常見中文字型，避免標題出現方塊字。
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "PingFang TC",
        "Noto Sans CJK TC",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


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


def compute_frame_diff(video_path: str) -> tuple[np.ndarray, float, int]:
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
        prev_gray = gray

        if frame_no % 100 == 0:
            update_progress("掃描 FrameDiff", frame_no + 1, total_frames, start_time)

    cap.release()
    processed = frame_no + 1
    update_progress("掃描 FrameDiff", processed, total_frames, start_time)

    return np.array(diffs, dtype=float), float(fps), processed


def load_frame_diff_csv(
    diff_csv_path: str,
    fallback_fps: float = 30.0,
) -> tuple[np.ndarray, float, int]:
    """從既有 frame diff CSV 載入 Difference_Score，跳過逐幀掃描。"""
    if not os.path.exists(diff_csv_path):
        raise FileNotFoundError(f"找不到 frame diff CSV: {diff_csv_path}")

    scores: list[float] = []
    times: list[float] = []
    frames: list[int] = []

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
            if time_text not in (None, ""):
                times.append(float(time_text))
            elif frame_text not in (None, ""):
                frames.append(int(float(frame_text)))
            else:
                frames.append(row_index)

    if not scores:
        raise ValueError("frame diff CSV 沒有可用資料（缺少 Difference_Score）")

    fps = fallback_fps
    if len(times) >= 2:
        deltas = np.diff(np.asarray(times, dtype=float))
        positive = deltas[deltas > 0]
        if positive.size > 0:
            fps = float(1.0 / np.median(positive))

    processed_frames = len(scores)
    return np.asarray(scores, dtype=float), float(fps), processed_frames


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


def pick_middle_frame(start_frame: int, end_frame: int) -> int:
    return (start_frame + end_frame) // 2


def collect_required_frames(segments: list[tuple[int, int]]) -> tuple[list[int], set[int]]:
    mids: list[int] = []
    required: set[int] = set()

    for start_frame, end_frame in segments:
        mid = pick_middle_frame(start_frame, end_frame)
        mids.append(mid)
        required.add(mid)

    return mids, required


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
    mid_frames: list[int],
    frame_cache: dict[int, np.ndarray],
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []

    for mid in mid_frames:
        frame = cv2.resize(frame_cache[mid], COMPARE_SIZE, interpolation=cv2.INTER_AREA)
        seg_vec = frame.reshape(-1).astype(np.int16)
        vectors.append(seg_vec)

    return vectors


def compute_pairwise_and_cross_avg(vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    seg_count = len(vectors)
    pairwise = np.zeros((seg_count, seg_count), dtype=np.float64)

    if seg_count <= 1:
        return pairwise, np.zeros(seg_count, dtype=np.float64)

    total_pairs = seg_count * (seg_count - 1) // 2
    done_pairs = 0
    start_time = time.time()
    update_progress("跨片段比對", 0, total_pairs, start_time)

    for i in range(seg_count):
        vi = vectors[i]
        for j in range(i + 1, seg_count):
            vj = vectors[j]
            diff_sum = float(np.abs(vi - vj).sum())
            pairwise[i, j] = diff_sum
            pairwise[j, i] = diff_sum

            done_pairs += 1
            if done_pairs % 20 == 0 or done_pairs == total_pairs:
                update_progress("跨片段比對", done_pairs, total_pairs, start_time)

    row_sums = np.sum(pairwise, axis=1)
    cross_avg = row_sums / max(seg_count - 1, 1)
    return pairwise, cross_avg


def build_frame_level_vectors(
    mid_frames: list[int],
    frame_cache: dict[int, np.ndarray],
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    labels: list[str] = []
    vectors: list[np.ndarray] = []
    display_frames: list[np.ndarray] = []

    for seg_idx, frame_idx in enumerate(mid_frames, start=1):
        resized = cv2.resize(frame_cache[frame_idx], COMPARE_SIZE, interpolation=cv2.INTER_AREA)
        vectors.append(resized.reshape(-1).astype(np.int16))
        display_frames.append(frame_cache[frame_idx])
        labels.append(f"S{seg_idx} (F{frame_idx})")

    return labels, vectors, display_frames


def compute_frame_pairwise(vectors: list[np.ndarray]) -> np.ndarray:
    n = len(vectors)
    pairwise = np.zeros((n, n), dtype=np.float64)

    if n <= 1:
        return pairwise

    total_pairs = n * (n - 1) // 2
    done_pairs = 0
    start_time = time.time()
    update_progress("代表幀比對", 0, total_pairs, start_time)

    for i in range(n):
        vi = vectors[i]
        for j in range(i + 1, n):
            vj = vectors[j]
            diff_sum = float(np.abs(vi - vj).sum())
            pairwise[i, j] = diff_sum
            pairwise[j, i] = diff_sum

            done_pairs += 1
            if done_pairs % 50 == 0 or done_pairs == total_pairs:
                update_progress("代表幀比對", done_pairs, total_pairs, start_time)

    return pairwise


def print_top_frame_pairs(frame_pairwise: np.ndarray, labels: list[str], top_k: int = 10) -> None:
    n = frame_pairwise.shape[0]
    if n <= 1:
        return

    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(frame_pairwise[i, j]), i, j))

    if not pairs:
        return

    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    top_k = min(top_k, len(pairs_sorted))

    print("-" * 60)
    print(f"最相似 Top-{top_k} 代表幀 pair")
    for value, i, j in pairs_sorted[:top_k]:
        print(f"{labels[i]} <-> {labels[j]} : {value:.0f}")

    print("-" * 60)
    print(f"最不相似 Top-{top_k} 代表幀 pair")
    for value, i, j in pairs_sorted[-top_k:][::-1]:
        print(f"{labels[i]} <-> {labels[j]} : {value:.0f}")


def select_evenly_spaced_indices(total: int, max_segments: int) -> list[int]:
    if total <= 0:
        return []
    if max_segments <= 0:
        return []
    if total <= max_segments:
        return list(range(total))

    points = np.linspace(0, total - 1, max_segments)
    indices = sorted({int(round(p)) for p in points})

    while len(indices) < max_segments:
        for idx in range(total):
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= max_segments:
                break

    return sorted(indices[:max_segments])


def make_montage(
    mid_frames: list[int],
    frame_cache: dict[int, np.ndarray],
    shown_indices: list[int],
) -> np.ndarray:
    tile_h = 60
    tile_w = 108
    pad = 6

    n = len(shown_indices)
    cols = min(10, max(1, int(math.ceil(math.sqrt(n)))))
    rows = max(1, int(math.ceil(n / cols)))

    canvas_h = pad + rows * (tile_h + pad)
    canvas_w = pad + cols * (tile_w + pad)
    canvas = np.full((canvas_h, canvas_w), 24, dtype=np.uint8)

    for idx, seg_idx in enumerate(shown_indices):
        row = idx // cols
        col = idx % cols
        frame_idx = mid_frames[seg_idx]

        y0 = pad + row * (tile_h + pad)
        x0 = pad + col * (tile_w + pad)
        tile = cv2.resize(frame_cache[frame_idx], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

        label = f"S{seg_idx + 1}-F{frame_idx}"
        cv2.putText(
            canvas,
            label,
            (x0 + 4, y0 + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            245,
            1,
            cv2.LINE_AA,
        )

    return canvas


def visualize(
    pairwise: np.ndarray,
    cross_avg: np.ndarray,
    frame_pairwise: np.ndarray,
    frame_labels: list[str],
    display_frames: list[np.ndarray],
    mid_frames: list[int],
    frame_cache: dict[int, np.ndarray],
    shown_indices: list[int],
    output_path: str | None,
    show_window: bool,
    frame_threshold: float,
    total_candidates: int,
) -> None:
    if not shown_indices:
        raise RuntimeError("沒有可視覺化的候選片段")

    shown_pairwise = pairwise[np.ix_(shown_indices, shown_indices)]
    shown_cross_avg = cross_avg[shown_indices]
    labels = [f"S{i + 1}" for i in shown_indices]

    montage = make_montage(mid_frames, frame_cache, shown_indices)
    n = len(labels)

    fig = plt.figure(figsize=(14, 9), dpi=110)
    spec = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=[1.55, 1.0],
        height_ratios=[0.95, 1.0],
        figure=fig,
    )

    ax_montage = fig.add_subplot(spec[0, :])
    ax_heatmap = fig.add_subplot(spec[1, 0])
    ax_bar = fig.add_subplot(spec[1, 1])

    ax_montage.imshow(montage, cmap="gray", vmin=0, vmax=255)
    ax_montage.set_title("候選片段代表幀（每段 1 幀）")
    ax_montage.axis("off")

    im = ax_heatmap.imshow(shown_pairwise, cmap="YlOrRd")
    ax_heatmap.set_title("Pairwise Frame Diff（每段 1 幀）")

    tick_step = 1 if n <= 24 else (2 if n <= 50 else 5)
    tick_positions = np.arange(0, n, tick_step)
    tick_labels = [labels[i] for i in tick_positions]
    ax_heatmap.set_xticks(tick_positions)
    ax_heatmap.set_yticks(tick_positions)
    ax_heatmap.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax_heatmap.set_yticklabels(tick_labels, fontsize=8)

    if n <= 28:
        max_value = max(float(shown_pairwise.max()), 1e-9)
        for r in range(shown_pairwise.shape[0]):
            for c in range(shown_pairwise.shape[1]):
                value = shown_pairwise[r, c]
                color = "black" if value < max_value * 0.6 else "white"
                ax_heatmap.text(c, r, f"{value:.0f}", ha="center", va="center", color=color, fontsize=7)

    fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04, label="diff")

    y = np.arange(len(labels))
    ax_bar.barh(y, shown_cross_avg, color="#4F81BD")
    ax_bar.set_yticks(tick_positions)
    ax_bar.set_yticklabels(tick_labels, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("cross_diff_avg")
    ax_bar.set_title("Cross_Diff_Avg（於顯示子集合內）")

    if n <= 35:
        for yi, val in enumerate(shown_cross_avg):
            ax_bar.text(val, yi, f" {val:.0f}", va="center", fontsize=7)

    fig.suptitle(
        "Cross Diff Avg Visualization\n"
        f"FrameDiff Threshold={frame_threshold:.2f} | 候選總數={total_candidates} | 顯示={len(shown_indices)}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig2 = plt.figure(figsize=(12, 10), dpi=110)
    ax_frame_heatmap = fig2.add_subplot(111)
    im2 = ax_frame_heatmap.imshow(frame_pairwise, cmap="viridis")
    ax_frame_heatmap.set_title("代表幀 Frame-to-Frame Pairwise Diff")

    fn = len(frame_labels)
    ftick_step = 1 if fn <= 24 else (2 if fn <= 50 else 5)
    fticks = np.arange(0, fn, ftick_step)
    flabels = [frame_labels[i] for i in fticks]
    ax_frame_heatmap.set_xticks(fticks)
    ax_frame_heatmap.set_yticks(fticks)
    ax_frame_heatmap.set_xticklabels(flabels, rotation=75, ha="right", fontsize=7)
    ax_frame_heatmap.set_yticklabels(flabels, fontsize=7)

    if len(frame_labels) <= 24:
        max_frame_value = max(float(frame_pairwise.max()), 1e-9)
        for r in range(frame_pairwise.shape[0]):
            for c in range(frame_pairwise.shape[1]):
                value = frame_pairwise[r, c]
                color = "white" if value > max_frame_value * 0.55 else "black"
                ax_frame_heatmap.text(c, r, f"{value:.0f}", ha="center", va="center", color=color, fontsize=6)

    fig2.colorbar(im2, ax=ax_frame_heatmap, fraction=0.046, pad=0.04, label="diff")
    fig2.tight_layout()

    n_frames = len(display_frames)
    cols = 10
    rows = max(1, int(math.ceil(n_frames / cols)))
    fig3 = plt.figure(figsize=(14, max(5, rows * 1.6)), dpi=110)
    fig3.suptitle("代表幀大圖牆")

    for idx, (label, gray_frame) in enumerate(zip(frame_labels, display_frames), start=1):
        ax = fig3.add_subplot(rows, cols, idx)
        ax.imshow(gray_frame, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=7)
        ax.axis("off")

    fig3.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        stem, ext = os.path.splitext(output_path)
        ext = ext if ext else ".png"
        out1 = f"{stem}_overview{ext}"
        out2 = f"{stem}_frame_matrix{ext}"
        out3 = f"{stem}_frame_gallery{ext}"
        fig.savefig(out1, bbox_inches="tight")
        fig2.savefig(out2, bbox_inches="tight")
        fig3.savefig(out3, bbox_inches="tight")
        print(f"已輸出圖檔: {out1}")
        print(f"已輸出圖檔: {out2}")
        print(f"已輸出圖檔: {out3}")

    if show_window:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)
        plt.close(fig3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="視覺化 cross_diff_avg 計算")
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="輸入影片路徑（預設為目前目錄第一個 .mp4）",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=12,
        help="畫面最多顯示幾個候選片段",
    )
    parser.add_argument(
        "--output",
        default="",
        help="輸出圖檔路徑（例如 cross_diff_avg.png）；不填則僅顯示視窗",
    )
    parser.add_argument(
        "--diff-csv",
        default="",
        help="可選：既有 frame diff CSV；提供後可跳過逐幀計算",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不開視窗，只輸出圖檔",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plot_font()

    video_path = args.video_path if args.video_path else pick_default_video()
    output_path = args.output if args.output else None

    print("=" * 60)
    print("Cross_Diff_Avg 視覺化")
    print("=" * 60)
    print(f"影片: {video_path}")
    print(f"max_segments: {args.max_segments}")
    if args.diff_csv:
        print(f"frame diff CSV: {args.diff_csv}")

    if args.diff_csv:
        scores, fps, processed_frames = load_frame_diff_csv(args.diff_csv)
    else:
        scores, fps, processed_frames = compute_frame_diff(video_path)
    frame_threshold = find_frame_threshold_gmm(scores)

    is_low = scores < frame_threshold
    if is_low.size > 0:
        is_low[0] = False

    raw_segments = build_segments_from_mask(is_low)
    candidate_segments = merge_close_segments(raw_segments, MERGE_GAP_FRAMES)

    if not candidate_segments:
        raise RuntimeError("沒有找到候選片段，無法視覺化")

    shown_indices = select_evenly_spaced_indices(len(candidate_segments), args.max_segments)
    shown_segments = [candidate_segments[i] for i in shown_indices]

    mid_frames, required_frames = collect_required_frames(shown_segments)
    max_required = max(required_frames) if required_frames else 0
    frame_cache = load_required_gray_frames(video_path, required_frames, max_required)

    vectors = build_segment_vectors(mid_frames, frame_cache)
    pairwise, cross_avg = compute_pairwise_and_cross_avg(vectors)
    frame_labels, frame_vectors, display_frames = build_frame_level_vectors(mid_frames, frame_cache)
    frame_pairwise = compute_frame_pairwise(frame_vectors)
    print_top_frame_pairs(frame_pairwise, frame_labels, top_k=12)

    visualize(
        pairwise=pairwise,
        cross_avg=cross_avg,
        frame_pairwise=frame_pairwise,
        frame_labels=frame_labels,
        display_frames=display_frames,
        mid_frames=mid_frames,
        frame_cache=frame_cache,
        shown_indices=list(range(len(shown_segments))),
        output_path=output_path,
        show_window=not args.no_show,
        frame_threshold=frame_threshold,
        total_candidates=len(candidate_segments),
    )

    print("-" * 60)
    print(f"FPS: {fps:.3f}")
    print(f"總幀數: {processed_frames}")
    print(f"FrameDiff GMM 分界值: {frame_threshold:.2f}")
    print(f"原始片段數: {len(raw_segments)}")
    print(f"候選片段數(合併後): {len(candidate_segments)}")
    print(f"實際顯示片段數: {len(shown_segments)}")
    print("完成")


if __name__ == "__main__":
    main()
