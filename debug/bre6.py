"""
bre6.py

讀取影片，計算每一幀與前一幀的 Difference Score，
使用 Gaussian Mixture Model (GMM) 自動找 rally / non-rally 分界，
並輸出每幀是否低於分界（低動態）到 CSV。

用法:
    python bre6.py input.mp4 output.csv

若未提供參數:
    - input.mp4 會使用目前目錄第一個 .mp4
    - output.csv 會命名為 <影片檔名>_bre6.csv
"""

from __future__ import annotations

import csv
import glob
import os
import sys

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


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

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    order = np.argsort(means)
    m1, m2 = means[order]
    v1, v2 = variances[order]
    w1, w2 = weights[order]

    v1 = max(v1, 1e-12)
    v2 = max(v2, 1e-12)

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


def write_csv(
    output_csv: str,
    scores: np.ndarray,
    times: np.ndarray,
    threshold: float,
) -> None:
    is_low = scores < threshold
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Frame",
                "Time_Sec",
                "Difference_Score",
                "Threshold",
                "Is_Low",
                "Label",
            ]
        )
        for i, (t, s, low) in enumerate(zip(times, scores, is_low)):
            writer.writerow(
                [
                    i,
                    f"{t:.6f}",
                    f"{s:.6f}",
                    f"{threshold:.6f}",
                    int(low),
                    "rally_like_low" if low else "non_rally_like_high",
                ]
            )


def main() -> None:
    video_path = sys.argv[1] if len(sys.argv) > 1 else pick_default_video()
    output_csv = (
        sys.argv[2]
        if len(sys.argv) > 2
        else f"{os.path.splitext(os.path.basename(video_path))[0]}_bre6.csv"
    )

    print("=" * 60)
    print("bre6: FrameDiff + GMM 分界 + 每幀分類")
    print("=" * 60)
    print(f"影片: {video_path}")
    print(f"輸出: {output_csv}")

    scores, times, fps = compute_frame_diff(video_path)
    threshold = find_threshold_gmm(scores)
    write_csv(output_csv, scores, times, threshold)

    low_count = int(np.sum(scores < threshold))
    high_count = int(scores.size - low_count)
    duration_sec = float(times[-1]) if times.size else 0.0

    print("-" * 60)
    print(f"FPS: {fps:.3f}")
    print(f"總幀數: {scores.size}")
    print(f"總時長: {duration_sec/60:.1f} 分鐘")
    print(f"GMM 分界值: {threshold:.2f}")
    print(f"低於分界: {low_count} 幀 ({low_count / max(scores.size, 1) * 100:.1f}%)")
    print(f"高於分界: {high_count} 幀 ({high_count / max(scores.size, 1) * 100:.1f}%)")
    print("完成")


if __name__ == "__main__":
    main()
