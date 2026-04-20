"""
自動分析 frame difference CSV，偵測羽球比賽回合片段，
並將所有回合接合成單一輸出影片。

用法:
    python.exe bre4.py test3.mp4 frame_diff_analysis3.csv output4.mp4

若不提供參數，程式會自動在當前目錄尋找 .mp4 與 .csv 檔案。
"""

import cv2
import sys
import os
import glob
import subprocess
import tempfile
import shutil
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture

try:
    from scipy.signal import find_peaks as scipy_find_peaks
except Exception:
    scipy_find_peaks = None

# 降低 OpenCV/FFmpeg 在有破損 GOP 時輸出的非致命解碼警告噪音
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "err_detect;ignore_err")

# ─────────────────────────────────────────────
#  可調整參數 (Tunable Parameters)
# ─────────────────────────────────────────────

# 平滑視窗（秒）：對 diff score 做滑動中位數平滑，消除單幀雜訊
SMOOTH_WINDOW_SEC = 0.5

# 寬容間距（秒）：兩個低谷片段之間若間隔小於此值，視為同一回合合併
#   → 處理回合中偶發的其他鏡位 (e.g. 發球特寫)
MERGE_GAP_SEC = 1.5

# 最短回合長度（秒）：低於此長度的片段視為雜訊，直接丟棄
MIN_RALLY_SEC = 2.5

# 固定鏡頭驗證：比較片段代表幀與「基準鏡頭」的色彩直方圖相似度
#   若相似度低於此閾值，視為非固定鏡頭（例如回放、觀眾鏡頭），跳過
#   範圍 0~1，建議 0.5~0.8，設為 0 則停用驗證
CAMERA_SIMILARITY_THRESHOLD = 0.7

# 手動閾值（設為 None 則自動偵測）
#   若自動偵測效果不佳，可手動設定，例如：MANUAL_THRESHOLD = 5_000_000
MANUAL_THRESHOLD = None

# ─────────────────────────────────────────────

def find_threshold_gmm(scores):

    """
    用 Gaussian Mixture Model 自動找 rally / non-rally 分界
    """

    # log scale
    log_scores = np.log10(np.maximum(scores, 1))

    # reshape for sklearn
    X = log_scores.reshape(-1, 1)

    # fit GMM (兩個高斯)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        random_state=0
    )

    gmm.fit(X)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_

    # 排序 (確保第一個是rally)
    order = np.argsort(means)
    m1, m2 = means[order]
    v1, v2 = variances[order]
    w1, w2 = weights[order]

    s1 = np.sqrt(v1)
    s2 = np.sqrt(v2)

    # 解兩個Gaussian相等的位置
    a = 1/(2*v1) - 1/(2*v2)
    b = m2/v2 - m1/v1
    c = m1**2/(2*v1) - m2**2/(2*v2) + np.log((s2*w1)/(s1*w2))

    roots = np.roots([a, b, c])

    # 取落在兩個mean之間的root
    thresh_log = np.real(roots[(roots > m1) & (roots < m2)])[0]

    threshold = 10 ** thresh_log

    return threshold


def find_rally_segments(scores: np.ndarray, fps: float, threshold: float) -> list[tuple[int,int]]:
    """
    以平滑後的 diff score 找出所有低谷片段（潛在回合），
    合併相近片段並過濾過短片段。
    回傳 [(start_frame, end_frame), ...] 列表（含頭尾）。
    """
    smooth_win = max(1, int(fps * SMOOTH_WINDOW_SEC))
    smoothed = pd.Series(scores).rolling(smooth_win, center=True, min_periods=1).median().values

    merge_gap   = int(fps * MERGE_GAP_SEC)
    min_rally   = int(fps * MIN_RALLY_SEC)

    in_rally = smoothed < threshold

    # 合併相近片段
    segments = []
    start = None
    gap_count = 0

    for i, flag in enumerate(in_rally):
        if flag:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                gap_count += 1
                if gap_count > merge_gap:
                    segments.append((start, i - gap_count))
                    start = None
                    gap_count = 0

    if start is not None:
        segments.append((start, len(scores) - 1))

    # 過濾過短片段
    segments = [(s, e) for s, e in segments if (e - s) >= min_rally]
    return segments


def extract_representative_frame(cap: cv2.VideoCapture, frame_no: int) -> np.ndarray | None:
    """擷取指定幀，轉為 HSV 色彩直方圖（用於鏡頭相似度比較）"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def histogram_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """計算兩個直方圖的 Bhattacharyya 相似度（1=完全相同，0=完全不同）"""
    corr = cv2.compareHist(
        h1.reshape(-1, 1).astype(np.float32),
        h2.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    return float(corr)  # -1~1，越高越相似


def verify_rally_segments(
    cap: cv2.VideoCapture,
    segments: list[tuple[int,int]],
    scores: np.ndarray,
    fps: float,
) -> list[tuple[int,int]]:
    """
    驗證片段是否為固定鏡頭回合畫面：
    1. 計算每個片段代表幀（中間幀）的色彩直方圖。
    2. 找出最長的幾個片段作為「基準鏡頭」。
    3. 若某片段與基準鏡頭的相似度低於閾值，視為非回合（例如廣告、觀眾席），排除。
    """
    if CAMERA_SIMILARITY_THRESHOLD <= 0 or not segments:
        return segments

    print("[驗證] 正在比對片段鏡頭相似度...")

    # 取得每個片段的代表幀直方圖
    hists = []
    for s, e in segments:
        mid = (s + e) // 2
        h = extract_representative_frame(cap, mid)
        hists.append(h)

    # 以最長的前 5 個片段作為基準（最可能是回合）
    lengths = [e - s for s, e in segments]
    top_n = min(5, len(segments))
    anchor_indices = sorted(np.argsort(lengths)[-top_n:])
    anchor_hists = [hists[i] for i in anchor_indices if hists[i] is not None]

    if not anchor_hists:
        return segments

    anchor_mean = np.mean(anchor_hists, axis=0)

    # 過濾
    verified = []
    for i, (s, e) in enumerate(segments):
        if hists[i] is None:
            continue
        sim = histogram_similarity(anchor_mean, hists[i])
        duration = (e - s) / fps
        if sim >= CAMERA_SIMILARITY_THRESHOLD:
            verified.append((s, e))
            print(f"  ✓ Frame {s:6d}-{e:6d} ({duration:.1f}s)  相似度={sim:.3f}")
        else:
            print(f"  ✗ Frame {s:6d}-{e:6d} ({duration:.1f}s)  相似度={sim:.3f} → 排除")

    return verified


def resolve_ffmpeg_executable() -> str | None:
    """回傳可用的 ffmpeg 執行檔路徑；找不到則回傳 None。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    # 常見 Windows 手動安裝路徑
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p

    # WinGet 常見安裝路徑（使用者層級安裝）
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        winget_patterns = [
            os.path.join(
                local_appdata,
                "Microsoft",
                "WinGet",
                "Packages",
                "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
                "ffmpeg-*",
                "bin",
                "ffmpeg.exe",
            ),
            os.path.join(
                local_appdata,
                "Microsoft",
                "WinGet",
                "Packages",
                "*FFmpeg*",
                "*",
                "bin",
                "ffmpeg.exe",
            ),
        ]
        for pattern in winget_patterns:
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[-1]

    return None


def resolve_ffmpeg_video_codec_args(ffmpeg_exe: str) -> list[str]:
    """
    依據目前 ffmpeg 可用編碼器回傳最合適的視訊參數。
    - 優先使用 libx264（品質/壓縮率最佳）
    - 其次 libopenh264
    - 最後退回內建 mpeg4（相容性最高）
    """
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        encoders_text = (result.stdout or "") + "\n" + (result.stderr or "")
    except Exception:
        encoders_text = ""

    if "libx264" in encoders_text:
        return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    if "libopenh264" in encoders_text:
        return ["-c:v", "libopenh264", "-b:v", "5M"]
    return ["-c:v", "mpeg4", "-q:v", "2"]


def write_opencv_concat(
    video_path: str,
    segments: list[tuple[int, int]],
    fps: float,
    output_path: str,
) -> bool:
    """不依賴 ffmpeg，直接用 OpenCV 將片段逐幀寫出。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片: {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = src_fps if src_fps and src_fps > 0 else fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"[錯誤] 無法建立輸出影片: {output_path}")
        return False

    print(f"\n[輸出] 偵測不到 ffmpeg，改用 OpenCV 逐幀輸出 → {output_path}")
    total_frames = sum((e - s + 1) for s, e in segments)
    written = 0

    try:
        for idx, (s, e) in enumerate(segments):
            cap.set(cv2.CAP_PROP_POS_FRAMES, s)
            for _ in range(s, e + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                written += 1

            print(
                f"  [{idx+1}/{len(segments)}] {s}~{e}  "
                f"進度 {written}/{total_frames} 幀",
                end="\r",
            )

        print()
        return written > 0
    finally:
        writer.release()
        cap.release()


def get_video_fps(video_path: str) -> float:
    """從影片檔案直接讀取實際 FPS，避免從 CSV 估算造成誤差。"""
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if actual_fps and actual_fps > 0:
        return actual_fps
    return 30.0


def write_ffmpeg_concat(video_path: str, segments: list[tuple[int,int]],
                        fps: float, output_path: str) -> bool:
    """
    使用 ffmpeg concat demuxer 將片段接合（需重新編碼以支援任意切割點）。
    先將每個片段裁切為暫存檔，再串接輸出。

    修正：
    - 使用影片實際 FPS（而非從 CSV 估算），避免時間偏移
    - -ss 放在 -i 之後做精準 frame-accurate seek（畫面才不會跑掉）
    """
    ffmpeg_exe = resolve_ffmpeg_executable()
    if ffmpeg_exe is None:
        return write_opencv_concat(video_path, segments, fps, output_path)

    video_codec_args = resolve_ffmpeg_video_codec_args(ffmpeg_exe)
    print(f"[FFmpeg] 視訊編碼參數: {' '.join(video_codec_args)}")

    # ★ 用影片本身的 FPS 換算秒數，不用 CSV 估算值
    actual_fps = get_video_fps(video_path)
    print(f"[FPS] 影片實際 FPS={actual_fps:.4f}  (CSV 估算={fps:.4f})")

    tmp_dir = tempfile.mkdtemp(prefix="rally_clips_")
    clip_paths = []

    try:
        print(f"\n[輸出] 正在裁切 {len(segments)} 個片段...")
        for idx, (s, e) in enumerate(segments):
            start_sec = s / actual_fps
            duration_sec = (e - s) / actual_fps
            clip_path = os.path.join(tmp_dir, f"clip_{idx:04d}.mp4")

            # ★ -ss / -t 放在 -i 之後 → frame-accurate seek，畫面精準不跑掉
            #   代價是速度稍慢，但對於已知區間的短片段影響不大
            cmd = [
                ffmpeg_exe, "-y",
                "-i", video_path,
                "-ss", f"{start_sec:.6f}",
                "-t", f"{duration_sec:.6f}",
                *video_codec_args,
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                clip_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  [警告] 片段 {idx} 裁切失敗: {result.stderr[-200:]}")
                continue
            clip_paths.append(clip_path)
            print(f"  [{idx+1}/{len(segments)}] {s}~{e} ({duration_sec:.1f}s)", end="\r")

        if not clip_paths:
            print("[錯誤] 沒有成功裁切任何片段")
            return False

        # 建立 concat list
        list_path = os.path.join(tmp_dir, "concat.txt")
        with open(list_path, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")

        print(f"\n[輸出] 正在合併 {len(clip_paths)} 個片段 → {output_path}")
        cmd = [
            ffmpeg_exe, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[錯誤] 合併失敗: {result.stderr[-300:]}")
            return False

        return True

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    # ── 參數處理 ──────────────────────────────────
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path   = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    if not video_path:
        mp4s = glob.glob("*.mp4")
        if not mp4s:
            print("[錯誤] 找不到 .mp4 檔案，請指定路徑")
            sys.exit(1)
        video_path = mp4s[0]

    if not csv_path:
        csvs = glob.glob("*.csv") + glob.glob("frame_diff*.csv")
        if not csvs:
            print("[錯誤] 找不到 .csv 檔案，請指定路徑")
            sys.exit(1)
        csv_path = csvs[0]

    if not output_path:
        base = os.path.splitext(video_path)[0]
        output_path = f"{base}_rallies.mp4"

    print("=" * 55)
    print("  羽球回合剪輯程式")
    print("=" * 55)
    print(f"  影片: {video_path}")
    print(f"  CSV : {csv_path}")
    print(f"  輸出: {output_path}")
    print("-" * 55)

    # ── 讀取 CSV ──────────────────────────────────
    df = pd.read_csv(csv_path)
    scores = df['Difference_Score'].values.astype(float)
    times  = df['Time_Sec'].values.astype(float)

    # 估算 FPS（從時間欄位）
    diffs = np.diff(times[times > 0])
    fps = 1.0 / np.median(diffs[diffs > 0]) if len(diffs) > 0 else 30.0
    print(f"  FPS  : {fps:.2f}")
    print(f"  總幀數: {len(scores):,}")
    print(f"  時長  : {times[-1]/60:.1f} 分鐘")

    # ── 閾值偵測 ──────────────────────────────────
    if MANUAL_THRESHOLD is not None:
        threshold = float(MANUAL_THRESHOLD)
        print(f"[閾值] 手動設定: {threshold/1e6:.2f}M")
    else:
        threshold = find_threshold_gmm(scores)

    # ── 找回合片段 ────────────────────────────────
    print(f"\n[偵測] 平滑視窗={SMOOTH_WINDOW_SEC}s  合併間距={MERGE_GAP_SEC}s  最短={MIN_RALLY_SEC}s")
    segments = find_rally_segments(scores, fps, threshold)
    print(f"[偵測] 初步找到 {len(segments)} 個片段")

    # ── 開啟影片做鏡頭驗證 ───────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片: {video_path}")
        sys.exit(1)

    segments = verify_rally_segments(cap, segments, scores, fps)
    cap.release()

    # ── 統計報告 ──────────────────────────────────
    total_rally_frames = sum(e - s for s, e in segments)
    total_rally_sec = total_rally_frames / fps
    print(f"\n[結果] 有效回合片段: {len(segments)} 個")
    print(f"[結果] 回合總時長  : {total_rally_sec:.1f}s = {total_rally_sec/60:.1f} 分鐘")
    print(f"[結果] 佔原片比例  : {total_rally_frames/len(scores)*100:.1f}%")
    print(f"\n{'#':>4}  {'起始幀':>8}  {'結束幀':>8}  {'時長(s)':>8}  {'起始時間':>10}")
    print("-" * 50)
    for i, (s, e) in enumerate(segments):
        start_time = s / fps
        dur = (e - s) / fps
        m, sec = divmod(start_time, 60)
        print(f"{i+1:>4}  {s:>8d}  {e:>8d}  {dur:>8.1f}  {int(m):02d}:{sec:05.2f}")

    # ── 輸出影片 ──────────────────────────────────
    if not segments:
        print("\n[警告] 沒有找到任何有效回合片段，程式結束。")
        sys.exit(0)

    ok = write_ffmpeg_concat(video_path, segments, fps, output_path)
    if ok:
        size_mb = os.path.getsize(output_path) / 1e6
        print(f"\n✅ 輸出完成: {output_path}  ({size_mb:.1f} MB)")
    else:
        print("\n❌ 輸出失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()