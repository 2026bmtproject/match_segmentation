#!/usr/bin/env python3
"""
依照 CSV 片段資訊剪輯 MP4，輸出合併後影片。

特性:
- 支援 Start_Sec/End_Sec 或 Start_Frame/End_Frame 欄位
- 平行切片，加速大量片段處理
- 自動偵測可用編碼器（優先 GPU）
- 兩種模式：
  - fast: 速度優先（可能非逐幀精準）
  - accurate: 精準優先（較慢）

用法:
    python fast_csv_cutter.py test2.mp4 test2_c4_segments_min2s.csv test2_c4-2.mp4

範例:
    python fast_csv_cutter.py test2.mp4 test2_c4_segments_min2s.csv test2_c4-2.mp4 --mode fast --workers 8
    python fast_csv_cutter.py test2.mp4 test2_c4_segments_min2s.csv test2_c4-2.mp4 --mode accurate --workers 4
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Segment:
    index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("找不到 ffmpeg，請先安裝並加入 PATH。")
    return ffmpeg


def find_ffprobe() -> str:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe
    ffmpeg_dir = Path(find_ffmpeg()).parent
    candidate = ffmpeg_dir / "ffprobe.exe"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("找不到 ffprobe，請確認 ffmpeg 套件完整安裝。")


def detect_video_encoder(ffmpeg: str) -> tuple[str, bool]:
    result = run_cmd([ffmpeg, "-hide_banner", "-encoders"])
    text = ((result.stdout or "") + "\n" + (result.stderr or "")).lower()

    for enc in ("h264_nvenc", "h264_qsv", "h264_amf"):
        if enc in text:
            return enc, True

    if "libx264" in text:
        return "libx264", False

    return "mpeg4", False


def get_video_fps(ffprobe: str, video_path: str) -> float:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    result = run_cmd(cmd)
    rate = (result.stdout or "").strip()
    if not rate:
        return 30.0

    if "/" in rate:
        n, d = rate.split("/", 1)
        try:
            n_f = float(n)
            d_f = float(d)
            if d_f > 0:
                return n_f / d_f
        except ValueError:
            return 30.0

    try:
        return float(rate)
    except ValueError:
        return 30.0


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_segments(csv_path: str, fps: float) -> list[Segment]:
    segments: list[Segment] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = set(reader.fieldnames or [])

        has_sec = {"Start_Sec", "End_Sec"}.issubset(required)
        has_frame = {"Start_Frame", "End_Frame"}.issubset(required)

        if not has_sec and not has_frame:
            raise ValueError(
                "CSV 需要包含 Start_Sec/End_Sec 或 Start_Frame/End_Frame 欄位。"
            )

        for idx, row in enumerate(reader, start=1):
            start_sec = parse_float(row.get("Start_Sec")) if has_sec else None
            end_sec = parse_float(row.get("End_Sec")) if has_sec else None

            if start_sec is None or end_sec is None:
                if not has_frame:
                    continue
                start_frame = parse_float(row.get("Start_Frame"))
                end_frame = parse_float(row.get("End_Frame"))
                if start_frame is None or end_frame is None:
                    continue
                start_sec = start_frame / fps
                end_sec = end_frame / fps

            if end_sec <= start_sec:
                continue

            segments.append(Segment(index=idx, start_sec=start_sec, end_sec=end_sec))

    if not segments:
        raise ValueError("CSV 沒有可用片段（請檢查時間欄位是否有效）。")

    return segments


def build_extract_cmd(
    ffmpeg: str,
    video_path: str,
    segment: Segment,
    out_path: str,
    mode: str,
    video_encoder: str,
) -> list[str]:
    if mode == "fast":
        # 速度優先：-ss 放在 -i 前方 + stream copy。
        # 這裡使用 -t (duration) 而非 -to (absolute end time)，
        # 可避免大量短片段時因時間基準差異造成總長被放大。
        return [
            ffmpeg,
            "-y",
            "-ss",
            f"{segment.start_sec:.6f}",
            "-t",
            f"{segment.duration_sec:.6f}",
            "-i",
            video_path,
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            out_path,
        ]

    video_args: list[str]
    if video_encoder == "libx264":
        video_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    elif video_encoder in ("h264_nvenc", "h264_qsv", "h264_amf"):
        video_args = ["-c:v", video_encoder, "-preset", "fast"]
    else:
        video_args = ["-c:v", "mpeg4", "-q:v", "3"]

    # 精準模式：-ss 放在 -i 後方，並重編碼確保精準切點。
    return [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-ss",
        f"{segment.start_sec:.6f}",
        "-t",
        f"{segment.duration_sec:.6f}",
        *video_args,
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        "-avoid_negative_ts",
        "make_zero",
        out_path,
    ]


def cut_one_segment(
    ffmpeg: str,
    video_path: str,
    segment: Segment,
    out_dir: str,
    mode: str,
    video_encoder: str,
) -> tuple[bool, str, str]:
    clip_name = f"clip_{segment.index:04d}.mp4"
    clip_path = os.path.join(out_dir, clip_name)

    cmd = build_extract_cmd(
        ffmpeg=ffmpeg,
        video_path=video_path,
        segment=segment,
        out_path=clip_path,
        mode=mode,
        video_encoder=video_encoder,
    )

    result = run_cmd(cmd)
    ok = result.returncode == 0
    tail = (result.stderr or "").strip()[-300:]

    if mode == "fast" and not ok:
        fallback_cmd = build_extract_cmd(
            ffmpeg=ffmpeg,
            video_path=video_path,
            segment=segment,
            out_path=clip_path,
            mode="accurate",
            video_encoder=video_encoder,
        )
        fallback_result = run_cmd(fallback_cmd)
        ok = fallback_result.returncode == 0
        tail = (fallback_result.stderr or "").strip()[-300:]

    return ok, clip_name, tail


def concat_clips(ffmpeg: str, clip_paths: Iterable[str], output_path: str, mode: str) -> None:
    clip_list = list(clip_paths)
    if not clip_list:
        raise RuntimeError("沒有可合併的片段。")

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as f:
        concat_file = f.name
        for p in clip_list:
            escaped = p.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    try:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
        ]

        # accurate 模式的每段已重編碼為一致格式，concat 時可 copy 提升速度。
        if mode == "accurate":
            cmd += ["-c", "copy", "-movflags", "+faststart", output_path]
        else:
            # fast 模式可能遇到少數段落時間戳差異，這裡重編碼一次保證穩定輸出。
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", output_path]

        result = run_cmd(cmd)
        if result.returncode != 0:
            raise RuntimeError("合併片段失敗:\n" + (result.stderr or "")[-500:])
    finally:
        try:
            os.remove(concat_file)
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="依照 CSV 片段資訊快速剪輯 MP4")
    parser.add_argument("input_mp4", help="輸入影片路徑 (.mp4)")
    parser.add_argument("input_csv", help="片段 CSV 路徑")
    parser.add_argument("output_mp4", help="輸出影片路徑 (.mp4)")
    parser.add_argument(
        "--mode",
        choices=["fast", "accurate"],
        default="fast",
        help="fast: 速度優先；accurate: 切點精準優先",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="平行切片工作數（預設為 CPU 核心數的一半）",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留中間片段，方便除錯",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_mp4 = os.path.abspath(args.input_mp4)
    input_csv = os.path.abspath(args.input_csv)
    output_mp4 = os.path.abspath(args.output_mp4)

    if not os.path.isfile(input_mp4):
        print(f"[錯誤] 找不到影片: {input_mp4}")
        return 1
    if not os.path.isfile(input_csv):
        print(f"[錯誤] 找不到 CSV: {input_csv}")
        return 1

    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe()

    fps = get_video_fps(ffprobe, input_mp4)
    segments = read_segments(input_csv, fps)

    encoder, using_gpu = detect_video_encoder(ffmpeg)

    print(f"[資訊] ffmpeg = {ffmpeg}")
    print(f"[資訊] 模式 = {args.mode}")
    print(f"[資訊] workers = {args.workers}")
    print(f"[資訊] 影片 fps = {fps:.4f}")
    print(f"[資訊] 片段數 = {len(segments)}")
    print(f"[資訊] 編碼器 = {encoder} (gpu={using_gpu})")

    Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)

    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="csv_cutter_")
    tmp_dir = tmp_dir_obj.name

    success_paths: list[str] = []
    failed = 0

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    cut_one_segment,
                    ffmpeg,
                    input_mp4,
                    segment,
                    tmp_dir,
                    args.mode,
                    encoder,
                ): segment
                for segment in segments
            }

            done = 0
            total = len(futures)

            for future in as_completed(futures):
                segment = futures[future]
                done += 1
                ok, clip_name, err_tail = future.result()
                clip_path = os.path.join(tmp_dir, clip_name)

                if ok and os.path.exists(clip_path):
                    success_paths.append(clip_path)
                    print(f"[{done:03d}/{total:03d}] OK   {clip_name}")
                else:
                    failed += 1
                    print(f"[{done:03d}/{total:03d}] FAIL {clip_name}")
                    if err_tail:
                        print(f"                {err_tail}")

        # 依原 CSV 順序合併
        success_paths.sort()
        concat_clips(ffmpeg, success_paths, output_mp4, args.mode)

        total_duration = sum(s.duration_sec for s in segments)
        print("\n=== 完成 ===")
        print(f"輸出影片: {output_mp4}")
        print(f"成功片段: {len(success_paths)} / {len(segments)}")
        print(f"失敗片段: {failed}")
        print(f"合計目標長度(秒): {total_duration:.3f}")

    finally:
        if args.keep_temp:
            print(f"[資訊] 暫存片段保留於: {tmp_dir}")
            tmp_dir_obj.cleanup = lambda: None
        else:
            tmp_dir_obj.cleanup()

    return 0 if success_paths else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[錯誤] {e}")
        sys.exit(1)
