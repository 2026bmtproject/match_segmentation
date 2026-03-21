#!/usr/bin/env python3
"""
bre4.py 的平行版本 - 用多個 ffmpeg 進程同時裁切片段，加快輸出速度。

用法:
    python bre4_parallel.py [video.mp4] [analysis.csv] [output.mp4] [--workers N]

差異：
- 使用執行緒池平行裁切片段（預設 = CPU 核心數）
- 保留所有音視頻同步邏輯（frame-accurate seek、音頻重編碼、時間戳修正）
- 最後用 concat 組合所有片段

# 預設（自動核心數）
python bre4_parallel.py TTYvsASY.mp4 frame_diff.csv ttyvasy_parallel.mp4

# 自訂 8 個平行工作
python bre4_parallel.py TTYvsASY.mp4 frame_diff.csv ttyvasy_parallel.mp4 --workers 8

# 4 個（節省 VRAM）
python bre4_parallel.py TTYvsASY.mp4 frame_diff.csv ttyvasy_parallel.mp4 --workers 4

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
import time

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

# 從 bre4.py 匯入共用函式
import bre4


def detect_gpu_memory() -> dict:
    """
    用 nvidia-smi 偵測可用 GPU 及其記憶體。
    回傳 {gpu_id: free_memory_mb, ...} 或 {} 若無 GPU。
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=5,
        )
        if result.returncode != 0:
            return {}
        
        gpu_dict = {}
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    gpu_id = int(parts[0].strip())
                    memory_mb = int(float(parts[1].strip()))
                    gpu_dict[gpu_id] = memory_mb
                except ValueError:
                    pass
        return gpu_dict
    except Exception as e:
        print(f"[GPU] 偵測失敗: {e}")
        return {}


def select_best_gpu(gpu_dict: dict) -> int | None:
    """
    選擇可用記憶體最多的 GPU。
    若無 GPU，回傳 None。
    """
    if not gpu_dict:
        return None
    best_gpu = max(gpu_dict.items(), key=lambda x: x[1])
    return best_gpu[0]


def detect_gpu_encoding_support(ffmpeg_exe: str, gpu_id: int | None = None) -> str | None:
    """
    檢查 ffmpeg 是否支援 h264_nvenc（GPU 編碼）。
    若支援且指定了 GPU，回傳 "h264_nvenc"；否則回傳 None。
    """
    if gpu_id is None:
        return None
    
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=5,
        )
        encoders_text = (result.stdout or "") + "\n" + (result.stderr or "")
        if "h264_nvenc" in encoders_text:
            return "h264_nvenc"
    except Exception:
        pass
    
    return None


def clip_with_ffmpeg_parallel(
    ffmpeg_exe: str,
    video_path: str,
    start_sec: float,
    duration_sec: float,
    video_codec_args: list,
    out_path: str,
    clip_index: int,
    total_clips: int,
    gpu_id: int | None = None,
) -> Tuple[bool, str, int]:
    """
    使用 ffmpeg 裁切單一片段。
    保留所有音視頻同步邏輯：
    - Frame-accurate seek (-ss / -t 放在 -i 之後)
    - 音頻重新編碼為 AAC 以確保同步
    - 時間戳修正 (-avoid_negative_ts make_zero)
    - 若指定 GPU，設定 CUDA_VISIBLE_DEVICES
    """
    # ★ 關鍵：-ss / -t 放在 -i 之後 → frame-accurate seek，音視同步精準
    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-ss", f"{start_sec:.6f}",
        "-t", f"{duration_sec:.6f}",
        *video_codec_args,
        "-c:a", "aac",  # ★ 重新編碼音頻以確保與視訊同步
        "-avoid_negative_ts", "make_zero",  # ★ 修正時間戳
        out_path
    ]
    
    try:
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=300,
            env=env,
        )
        ok = result.returncode == 0
        err_msg = result.stderr[-200:] if result.stderr else ""
        return ok, err_msg, clip_index
    except subprocess.TimeoutExpired:
        return False, "timeout (300s)", clip_index
    except Exception as e:
        return False, str(e)[-200:], clip_index


def clip_segments_parallel(
    ffmpeg_exe: str,
    video_path: str,
    segments: list,
    actual_fps: float,
    video_codec_args: list,
    tmp_dir: str,
    num_workers: int = None,
    gpu_id: int | None = None,
) -> Tuple[list, int]:
    """
    使用執行緒池平行裁切所有片段。
    
    Args:
        ffmpeg_exe: ffmpeg 執行檔路徑
        video_path: 源影片路徑
        segments: [(start_frame, end_frame), ...] 列表
        actual_fps: 影片實際 FPS
        video_codec_args: 視訊編碼參數
        tmp_dir: 暫存目錄
        num_workers: 並行度（預設 = CPU 核心數）
        gpu_id: GPU ID（若使用 GPU 編碼）
    
    Returns:
        (成功的 clip 路徑列表, 失敗數)
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    
    gpu_mode = "GPU (h264_nvenc)" if gpu_id is not None else "CPU"
    print(f"[輸出] 平行裁切模式，workers={num_workers}，編碼器={gpu_mode}")
    print(f"[輸出] 正在裁切 {len(segments)} 個片段...")
    
    clip_paths = []
    failed_count = 0
    completed = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        
        # 提交所有任務
        for idx, (s, e) in enumerate(segments):
            start_sec = s / actual_fps
            duration_sec = (e - s) / actual_fps
            clip_path = os.path.join(tmp_dir, f"clip_{idx:04d}.mp4")
            
            future = executor.submit(
                clip_with_ffmpeg_parallel,
                ffmpeg_exe=ffmpeg_exe,
                video_path=video_path,
                start_sec=start_sec,
                duration_sec=duration_sec,
                video_codec_args=video_codec_args,
                out_path=clip_path,
                clip_index=idx,
                total_clips=len(segments),
                gpu_id=gpu_id,
            )
            futures[future] = (idx, s, e, duration_sec, clip_path)
        
        # 處理完成的任務
        for future in as_completed(futures):
            idx, s, e, duration_sec, clip_path = futures[future]
            completed += 1
            
            try:
                ok, err_msg, _ = future.result()
                if ok:
                    clip_paths.append((idx, clip_path))
                    status = "OK"
                    print(
                        f"  [{completed:3d}/{len(segments):3d}] {status} "
                        f"Frame {s:6d}-{e:6d} ({duration_sec:.2f}s)",
                        end="\r"
                    )
                else:
                    failed_count += 1
                    print(
                        f"  [{completed:3d}/{len(segments):3d}] FAIL "
                        f"Frame {s:6d}-{e:6d} ({duration_sec:.2f}s)"
                    )
                    if err_msg:
                        print(f"      Error: {err_msg}")
            except Exception as e:
                failed_count += 1
                print(
                    f"  [{completed:3d}/{len(segments):3d}] FAIL "
                    f"Frame {s:6d}-{e:6d} - Exception: {str(e)[-100:]}"
                )
        
        elapsed = time.time() - start_time
        print()
        print(f"[輸出] 裁切完成：成功 {len(clip_paths)}/{len(segments)}，耗時 {elapsed:.1f}s")
    
    # 按 index 排序以保持順序
    clip_paths.sort(key=lambda x: x[0])
    return [p for _, p in clip_paths], failed_count


def write_concat_list(clip_paths: list, list_path: str) -> None:
    """建立 ffmpeg concat 清單。"""
    with open(list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")


def merge_segments_with_concat(
    ffmpeg_exe: str,
    clip_paths: list,
    output_path: str,
    tmp_dir: str,
) -> bool:
    """
    使用 ffmpeg concat demuxer 組合所有片段。
    保留音視同步：使用 -c copy 避免重新編碼（片段已是正確格式）。
    """
    if not clip_paths:
        print("[錯誤] 沒有可合併的片段")
        return False
    
    list_path = os.path.join(tmp_dir, "concat.txt")
    write_concat_list(clip_paths, list_path)
    
    print(f"[輸出] 正在合併 {len(clip_paths)} 個片段 → {output_path}")
    
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",  # ★ 複製不重新編碼 - 保留音視同步
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        print(f"[錯誤] 合併失敗: {result.stderr[-300:]}")
        return False
    
    return True


def write_ffmpeg_concat_parallel(
    video_path: str,
    segments: list,
    fps: float,
    output_path: str,
    num_workers: int = None,
) -> bool:
    """
    平行版本的 ffmpeg 組合流程。
    1. 偵測 GPU 並選擇記憶體最大的那張
    2. 若支援 h264_nvenc，使用 GPU 編碼；否則用 CPU
    3. 平行裁切所有片段（多執行緒）
    4. 用 concat demuxer 組合
    """
    ffmpeg_exe = bre4.resolve_ffmpeg_executable()
    if ffmpeg_exe is None:
        print("[警告] 找不到 ffmpeg，改用 OpenCV 非平行輸出")
        return bre4.write_opencv_concat(video_path, segments, fps, output_path)
    
    # 偵測 GPU
    gpu_dict = detect_gpu_memory()
    selected_gpu = None
    gpu_encoder_args = None
    
    if gpu_dict:
        selected_gpu = select_best_gpu(gpu_dict)
        print(f"[GPU] 偵測到 {len(gpu_dict)} 張 GPU")
        for gpu_id, mem_mb in gpu_dict.items():
            marker = " ← 選擇" if gpu_id == selected_gpu else ""
            print(f"      GPU {gpu_id}: {mem_mb:,} MB 可用{marker}")
        
        # 檢查是否支援 h264_nvenc
        if detect_gpu_encoding_support(ffmpeg_exe, selected_gpu):
            print(f"[GPU] GPU {selected_gpu} 支援 h264_nvenc 硬體編碼")
            gpu_encoder_args = ["-c:v", "h264_nvenc", "-preset", "fast"]
    
    # 決定編碼器
    if gpu_encoder_args:
        video_codec_args = gpu_encoder_args
        print(f"[FFmpeg] 視訊編碼參數: {' '.join(video_codec_args)}")
    else:
        video_codec_args = bre4.resolve_ffmpeg_video_codec_args(ffmpeg_exe)
        print(f"[FFmpeg] 視訊編碼參數: {' '.join(video_codec_args)}")
    
    actual_fps = bre4.get_video_fps(video_path)
    print(f"[FPS] 影片實際 FPS={actual_fps:.4f}  (CSV 估算={fps:.4f})")
    
    tmp_dir = tempfile.mkdtemp(prefix="rally_clips_parallel_")
    
    try:
        # 平行裁切
        clip_paths, failed_count = clip_segments_parallel(
            ffmpeg_exe=ffmpeg_exe,
            video_path=video_path,
            segments=segments,
            actual_fps=actual_fps,
            video_codec_args=video_codec_args,
            tmp_dir=tmp_dir,
            num_workers=num_workers,
            gpu_id=selected_gpu,
        )
        
        if not clip_paths:
            print("[錯誤] 沒有成功裁切任何片段")
            return False
        
        # 合併
        ok = merge_segments_with_concat(
            ffmpeg_exe=ffmpeg_exe,
            clip_paths=clip_paths,
            output_path=output_path,
            tmp_dir=tmp_dir,
        )
        
        return ok
    
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    # ── 參數處理 ──────────────────────────────────
    num_workers = None
    
    # 檢查 --workers 參數
    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        if idx + 1 < len(sys.argv):
            try:
                num_workers = int(sys.argv[idx + 1])
                sys.argv.pop(idx + 1)
                sys.argv.pop(idx)
            except ValueError:
                pass
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None
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
    print("  羽球回合剪輯程式 (平行版)")
    print("=" * 55)
    print(f"  影片: {video_path}")
    print(f"  CSV : {csv_path}")
    print(f"  輸出: {output_path}")
    if num_workers:
        print(f"  Workers: {num_workers}")
    print("-" * 55)
    
    # ── 讀取 CSV ──────────────────────────────────
    df = pd.read_csv(csv_path)
    scores = df['Difference_Score'].values.astype(float)
    times = df['Time_Sec'].values.astype(float)
    
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
        threshold = bre4.find_threshold_gmm(scores)
    
    # ── 找回合片段 ────────────────────────────────
    print(f"\n[偵測] 平滑視窗={SMOOTH_WINDOW_SEC}s  合併間距={MERGE_GAP_SEC}s  最短={MIN_RALLY_SEC}s")
    segments = bre4.find_rally_segments(scores, fps, threshold)
    print(f"[偵測] 初步找到 {len(segments)} 個片段")
    
    # ── 開啟影片做鏡頭驗證 ───────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片: {video_path}")
        sys.exit(1)
    
    segments = bre4.verify_rally_segments(cap, segments, scores, fps)
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
    
    # ── 平行輸出影片 ──────────────────────────────
    if not segments:
        print("\n[警告] 沒有找到任何有效回合片段，程式結束。")
        sys.exit(0)
    
    ok = write_ffmpeg_concat_parallel(
        video_path, segments, fps, output_path, num_workers=num_workers
    )
    if ok:
        size_mb = os.path.getsize(output_path) / 1e6
        print(f"\n✅ 輸出完成: {output_path}  ({size_mb:.1f} MB)")
    else:
        print("\n❌ 輸出失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
