#!/usr/bin/env python3
"""
從 bre4.py 的回合偵測邏輯輸出切片版本。

用法:
    python bre4_slice.py [video.mp4] [analysis.csv] [output_dir]

預設:
    - video.mp4: 目前目錄第一個 .mp4
    - analysis.csv: 目前目錄第一個 .csv
    - output_dir: <video_basename>_rally_clips
"""

from __future__ import annotations

import csv
import glob
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

import bre4

# 降低 OpenCV/FFmpeg 在有破損 GOP 時輸出的非致命解碼警告噪音
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "err_detect;ignore_err")


def resolve_inputs() -> tuple[str, str, str]:
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    if not video_path:
        mp4s = sorted(glob.glob("*.mp4"))
        if not mp4s:
            raise FileNotFoundError("目前目錄找不到 .mp4 檔案")
        video_path = mp4s[0]

    if not csv_path:
        csvs = sorted(glob.glob("*.csv") + glob.glob("frame_diff*.csv"))
        if not csvs:
            raise FileNotFoundError("目前目錄找不到 .csv 檔案")
        csv_path = csvs[0]

    if not output_dir:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"{base}_rally_clips"

    return video_path, csv_path, output_dir


def detect_video_encoder(ffmpeg_exe: str) -> tuple[str, bool]:
    """
    依據可用編碼器回傳最合適的視訊編碼器。
    - 優先硬體加速（h264_nvenc, h264_qsv, h264_amf）
    - 其次 libx264（GPU-free）
    - 第三 libopenh264
    - 最後退回 mpeg4（相容性最高）
    """
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return "mpeg4", False

    text = (result.stdout + "\n" + result.stderr).lower()
    
    # 優先硬體加速
    for enc in ("h264_nvenc", "h264_qsv", "h264_amf"):
        if enc in text:
            return enc, True
    
    # 次選 libx264
    if "libx264" in text:
        return "libx264", False
    
    # 第三 libopenh264
    if "libopenh264" in text:
        return "libopenh264", False
    
    # 最後退回 mpeg4
    return "mpeg4", False


def build_video_codec_args(video_encoder: str) -> list[str]:
    """
    根據編碼器回傳對應的 ffmpeg 參數。
    """
    if video_encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    if video_encoder == "libopenh264":
        return ["-c:v", "libopenh264", "-b:v", "5M"]
    if video_encoder in ("h264_nvenc", "h264_qsv", "h264_amf"):
        return ["-c:v", video_encoder, "-preset", "fast"]
    # mpeg4 或其他
    return ["-c:v", video_encoder, "-q:v", "2"]


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if actual_fps and actual_fps > 0:
        return float(actual_fps)
    return 30.0


def clip_with_ffmpeg(
    ffmpeg_exe: str,
    video_path: str,
    start_sec: float,
    duration_sec: float,
    video_encoder: str,
    out_path: str,
) -> tuple[bool, str]:
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        video_path,
        "-ss",
        f"{start_sec:.6f}",
        "-t",
        f"{duration_sec:.6f}",
        *build_video_codec_args(video_encoder),
        "-c:a",
        "aac",
        "-avoid_negative_ts",
        "make_zero",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0, result.stderr[-300:]


def clip_with_opencv(
    video_path: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    out_path: str,
) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = src_fps if src_fps and src_fps > 0 else fps

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok = False
    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        ok = True

    writer.release()
    cap.release()
    return ok


def write_segments_to_dir(
    video_path: str,
    segments: list[tuple[int, int]],
    fps: float,
    output_dir: str,
) -> tuple[int, str]:
    os.makedirs(output_dir, exist_ok=True)
    actual_fps = get_video_fps(video_path)

    ffmpeg_exe = bre4.resolve_ffmpeg_executable()
    video_encoder = "opencv"
    using_gpu = False
    if ffmpeg_exe:
        video_encoder, using_gpu = detect_video_encoder(ffmpeg_exe)

    mode = "ffmpeg" if ffmpeg_exe else "opencv"
    print(f"[output] mode={mode}, dir={output_dir}")
    print(f"[output] actual_fps={actual_fps:.4f} (csv_fps={fps:.4f})")
    if ffmpeg_exe:
        print(f"[output] video encoder={video_encoder} (gpu={using_gpu})")

    rows: list[dict[str, str | int | float]] = []
    ok_count = 0

    for idx, (start_frame, end_frame) in enumerate(segments, start=1):
        start_sec = float(start_frame / actual_fps)
        duration_sec = float((end_frame - start_frame) / actual_fps)
        end_sec = start_sec + duration_sec
        clip_name = f"clip_{idx:04d}.mp4"
        out_path = os.path.join(output_dir, clip_name)

        if ffmpeg_exe:
            ok, err_tail = clip_with_ffmpeg(
                ffmpeg_exe=ffmpeg_exe,
                video_path=video_path,
                start_sec=start_sec,
                duration_sec=duration_sec,
                video_encoder=video_encoder,
                out_path=out_path,
            )
            if (not ok) and video_encoder != "libx264":
                ok, err_tail = clip_with_ffmpeg(
                    ffmpeg_exe=ffmpeg_exe,
                    video_path=video_path,
                    start_sec=start_sec,
                    duration_sec=duration_sec,
                    video_encoder="libx264",
                    out_path=out_path,
                )
        else:
            ok = clip_with_opencv(video_path, start_frame, end_frame, fps, out_path)
            err_tail = ""

        if ok:
            ok_count += 1
            rows.append(
                {
                    "clip": clip_name,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_sec": round(start_sec, 4),
                    "end_sec": round(end_sec, 4),
                    "duration_sec": round(duration_sec, 4),
                }
            )
            print(
                f"  [{idx:03d}/{len(segments):03d}] OK   {clip_name} "
                f"({duration_sec:.2f}s)"
            )
        else:
            print(f"  [{idx:03d}/{len(segments):03d}] FAIL {clip_name}")
            if err_tail:
                print(f"      ffmpeg: {err_tail.strip()}")

    index_csv = os.path.join(output_dir, "segments.csv")
    with open(index_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip",
                "start_frame",
                "end_frame",
                "start_sec",
                "end_sec",
                "duration_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return ok_count, index_csv


def main() -> None:
    try:
        video_path, csv_path, output_dir = resolve_inputs()
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        sys.exit(1)

    print("=" * 60)
    print("bre4 Rally Slice Exporter")
    print("=" * 60)
    print(f"video     : {video_path}")
    print(f"csv       : {csv_path}")
    print(f"output dir: {output_dir}")

    df = pd.read_csv(csv_path)
    scores = df["Difference_Score"].values.astype(float)
    times = df["Time_Sec"].values.astype(float)

    diffs = np.diff(times[times > 0])
    fps = 1.0 / np.median(diffs[diffs > 0]) if len(diffs) > 0 else 30.0

    if bre4.MANUAL_THRESHOLD is not None:
        threshold = float(bre4.MANUAL_THRESHOLD)
        print(f"[detect] manual threshold={threshold:.0f}")
    else:
        threshold = bre4.find_threshold_gmm(scores)
        print(f"[detect] auto threshold={threshold:.0f}")

    print(
        f"[detect] smooth={bre4.SMOOTH_WINDOW_SEC}s "
        f"merge_gap={bre4.MERGE_GAP_SEC}s min={bre4.MIN_RALLY_SEC}s"
    )
    segments = bre4.find_rally_segments(scores, fps, threshold)
    print(f"[detect] initial segments={len(segments)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] cannot open video: {video_path}")
        sys.exit(1)

    segments = bre4.verify_rally_segments(cap, segments, scores, fps)
    cap.release()
    print(f"[detect] verified segments={len(segments)}")

    if not segments:
        print("[warn] no valid rally segments found")
        sys.exit(0)

    ok_count, index_csv = write_segments_to_dir(video_path, segments, fps, output_dir)
    print("-" * 60)
    print(f"[done] exported clips: {ok_count}/{len(segments)}")
    print(f"[done] index csv     : {index_csv}")

    if ok_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
