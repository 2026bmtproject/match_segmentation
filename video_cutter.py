#!/usr/bin/env python3
"""
FFmpeg 影片剪輯工具
依照 CSV 檔案剪輯 MP4 影片，支援分段輸出或合併輸出
python video_cutter.py -v test2.mp4 -c r.csv -m merge -o merged.mp4
python video_cutter.py -v test2.mp4 -c r.csv -m separate -o ./output_clips
"""

import subprocess
import csv
import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path


# ──────────────────────────────────────────────
#  工具函式
# ──────────────────────────────────────────────

def check_ffmpeg():
    """確認 ffmpeg 是否已安裝"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[錯誤] 找不到 ffmpeg，請先安裝 ffmpeg 並確認已加入 PATH。")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt install ffmpeg")
        print("  Windows: https://ffmpeg.org/download.html")
        return False


def get_video_duration(video_path: str) -> float:
    """使用 ffprobe 取得影片長度（秒）"""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 失敗：{result.stderr.strip() or '無法取得影片長度'}")

    try:
        duration = float(result.stdout.strip())
    except ValueError as e:
        raise RuntimeError("ffprobe 回傳的影片長度格式錯誤") from e

    if duration <= 0:
        raise RuntimeError("影片長度小於等於 0，無法處理")
    return duration


def parse_csv(csv_path: str) -> list[dict]:
    """讀取 CSV 並回傳片段清單"""
    required = {"Start_Sec", "End_Sec"}
    segments = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])

        missing = required - headers
        if missing:
            raise ValueError(f"CSV 缺少必要欄位：{', '.join(missing)}")

        for i, row in enumerate(reader, start=1):
            try:
                seg = {
                    "index":         i,
                    "start_sec":     float(row["Start_Sec"]),
                    "end_sec":       float(row["End_Sec"]),
                    "duration_sec":  float(row.get("Duration_Sec") or
                                          float(row["End_Sec"]) - float(row["Start_Sec"])),
                    "start_frame":   row.get("Start_Frame", ""),
                    "end_frame":     row.get("End_Frame", ""),
                }
                if seg["end_sec"] <= seg["start_sec"]:
                    print(f"  [警告] 第 {i} 段 End_Sec ({seg['end_sec']}) <= Start_Sec ({seg['start_sec']})，略過。")
                    continue
                segments.append(seg)
            except ValueError as e:
                print(f"  [警告] 第 {i} 行解析失敗（{e}），略過。")

    return segments


def sec_to_ts(seconds: float) -> str:
    """秒數轉 HH:MM:SS.mmm 格式"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def run_ffmpeg(args: list[str], desc: str = "") -> bool:
    """執行 ffmpeg 指令，回傳是否成功"""
    if desc:
        print(f"  → {desc}")
    result = subprocess.run(
        args,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  [ffmpeg 錯誤]\n{result.stderr[-800:]}")
        return False
    return True


# ──────────────────────────────────────────────
#  模式 A：每段分開輸出
# ──────────────────────────────────────────────

def mode_separate(video_path: str, segments: list[dict], output_dir: str):
    """每個片段輸出為獨立的 MP4 檔案"""
    os.makedirs(output_dir, exist_ok=True)
    video_stem = Path(video_path).stem
    total = len(segments)
    success = 0

    print(f"\n[模式 A] 分段輸出 → 資料夾：{output_dir}")
    print(f"共 {total} 段\n")

    for seg in segments:
        idx   = seg["index"]
        start = seg["start_sec"]
        end   = seg["end_sec"]
        dur   = seg["duration_sec"]
        out   = os.path.join(output_dir, f"{video_stem}_seg{idx:04d}.mp4")

        print(f"  [{idx}/{total}] {sec_to_ts(start)} → {sec_to_ts(end)}  ({dur:.3f}s)")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", video_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            out
        ]

        if run_ffmpeg(cmd, f"輸出 {os.path.basename(out)}"):
            print(f"    ✓ 儲存：{out}")
            success += 1
        else:
            print(f"    ✗ 失敗：第 {idx} 段")

    print(f"\n完成：{success}/{total} 段輸出成功。")


# ──────────────────────────────────────────────
#  模式 B：合併成一個影片
# ──────────────────────────────────────────────

def mode_merge(video_path: str, segments: list[dict], output_path: str):
    """將所有片段合併為單一 MP4 檔案"""
    total = len(segments)
    print(f"\n[模式 B] 合併輸出 → {output_path}")
    print(f"共 {total} 段，先暫存再串接...\n")

    tmp_dir = tempfile.mkdtemp(prefix="ffmpeg_merge_")
    tmp_files = []

    try:
        # 步驟 1：每段暫存
        for seg in segments:
            idx   = seg["index"]
            start = seg["start_sec"]
            end   = seg["end_sec"]
            dur   = seg["duration_sec"]
            tmp   = os.path.join(tmp_dir, f"seg{idx:04d}.mp4")

            print(f"  [{idx}/{total}] {sec_to_ts(start)} → {sec_to_ts(end)}  ({dur:.3f}s)")

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", video_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                tmp
            ]

            if run_ffmpeg(cmd, f"暫存第 {idx} 段"):
                tmp_files.append(tmp)
                print(f"    ✓ 暫存成功")
            else:
                print(f"    ✗ 暫存失敗，略過第 {idx} 段")

        if not tmp_files:
            print("\n[錯誤] 沒有任何片段暫存成功，無法合併。")
            return

        # 步驟 2：建立 concat list
        list_file = os.path.join(tmp_dir, "concat_list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in tmp_files:
                f.write(f"file '{p}'\n")

        # 步驟 3：concat
        print(f"\n  正在合併 {len(tmp_files)} 段...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ]

        if run_ffmpeg(cmd, f"合併輸出 → {output_path}"):
            print(f"\n  ✓ 合併完成：{output_path}")
        else:
            print("\n  ✗ 合併失敗。")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def mode_inverse_merge(video_path: str, segments: list[dict], output_path: str):
    """將 CSV 片段全部刪除，只保留其補集並合併輸出"""
    print(f"\n[模式 C] 反向合併輸出（刪除 CSV 片段）→ {output_path}")

    try:
        video_duration = get_video_duration(video_path)
    except Exception as e:
        print(f"[錯誤] 讀取影片長度失敗：{e}")
        return

    # 先把片段正規化到影片範圍內，並排序。
    normalized = []
    for seg in segments:
        start = max(0.0, seg["start_sec"])
        end = min(video_duration, seg["end_sec"])
        if end > start:
            normalized.append((start, end))

    if normalized:
        normalized.sort(key=lambda x: x[0])
    else:
        print("\n  CSV 片段都不在影片範圍內，將保留整支影片。")

    # 合併重疊/相鄰的刪除片段。
    merged_remove = []
    for start, end in normalized:
        if not merged_remove:
            merged_remove.append([start, end])
            continue
        last_start, last_end = merged_remove[-1]
        if start <= last_end:
            merged_remove[-1][1] = max(last_end, end)
        else:
            merged_remove.append([start, end])

    # 取補集：保留區段。
    keep_segments = []
    cursor = 0.0
    for start, end in merged_remove:
        if start > cursor:
            keep_segments.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < video_duration:
        keep_segments.append((cursor, video_duration))

    if not keep_segments:
        print("\n[錯誤] CSV 片段已覆蓋整部影片，沒有可保留的區段。")
        return

    print(f"  影片總長：{video_duration:.3f}s")
    print(f"  刪除區段：{len(merged_remove)} 段")
    print(f"  保留區段：{len(keep_segments)} 段，先暫存再串接...\n")

    tmp_dir = tempfile.mkdtemp(prefix="ffmpeg_inverse_merge_")
    tmp_files = []

    try:
        for idx, (start, end) in enumerate(keep_segments, start=1):
            dur = end - start
            tmp = os.path.join(tmp_dir, f"keep{idx:04d}.mp4")

            print(f"  [{idx}/{len(keep_segments)}] {sec_to_ts(start)} → {sec_to_ts(end)}  ({dur:.3f}s)")

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", video_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                tmp
            ]

            if run_ffmpeg(cmd, f"暫存保留片段 {idx}"):
                tmp_files.append(tmp)
                print("    ✓ 暫存成功")
            else:
                print(f"    ✗ 暫存失敗，略過第 {idx} 段")

        if not tmp_files:
            print("\n[錯誤] 沒有任何保留片段暫存成功，無法合併。")
            return

        list_file = os.path.join(tmp_dir, "concat_list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in tmp_files:
                f.write(f"file '{p}'\n")

        print(f"\n  正在合併 {len(tmp_files)} 段...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ]

        if run_ffmpeg(cmd, f"反向合併輸出 → {output_path}"):
            print(f"\n  ✓ 反向合併完成：{output_path}")
        else:
            print("\n  ✗ 反向合併失敗。")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────
#  互動式選單（無引數時啟動）
# ──────────────────────────────────────────────

def interactive_mode():
    print("=" * 52)
    print("   FFmpeg 影片剪輯工具  v1.0")
    print("=" * 52)

    # 輸入影片
    while True:
        video = input("\n請輸入 MP4 影片路徑：").strip().strip("'\"")
        if os.path.isfile(video):
            break
        print("  [錯誤] 找不到檔案，請重新輸入。")

    # 輸入 CSV
    while True:
        csv_f = input("請輸入 CSV 檔案路徑：").strip().strip("'\"")
        if os.path.isfile(csv_f):
            break
        print("  [錯誤] 找不到檔案，請重新輸入。")

    # 讀取 CSV
    try:
        segments = parse_csv(csv_f)
    except Exception as e:
        print(f"\n[錯誤] 讀取 CSV 失敗：{e}")
        sys.exit(1)

    if not segments:
        print("\n[錯誤] CSV 中沒有有效片段。")
        sys.exit(1)

    print(f"\n  讀取到 {len(segments)} 個有效片段。")

    # 選擇模式
    print("\n請選擇輸出模式：")
    print("  1. 分段輸出（每個片段獨立 MP4）")
    print("  2. 合併輸出（所有片段合成一個 MP4）")
    print("  3. 反向合併（刪除 CSV 片段，只保留其餘畫面）")

    while True:
        choice = input("請輸入 1、2 或 3：").strip()
        if choice in ("1", "2", "3"):
            break
        print("  請輸入 1、2 或 3。")

    if choice == "1":
        default_dir = Path(video).stem + "_segments"
        out = input(f"輸出資料夾（預設：{default_dir}）：").strip() or default_dir
        mode_separate(video, segments, out)
    elif choice == "2":
        default_out = Path(video).stem + "_merged.mp4"
        out = input(f"輸出檔案（預設：{default_out}）：").strip() or default_out
        mode_merge(video, segments, out)
    else:
        default_out = Path(video).stem + "_inverse_merged.mp4"
        out = input(f"輸出檔案（預設：{default_out}）：").strip() or default_out
        mode_inverse_merge(video, segments, out)


# ──────────────────────────────────────────────
#  CLI 引數模式
# ──────────────────────────────────────────────

def cli_mode():
    parser = argparse.ArgumentParser(
        description="FFmpeg 影片剪輯工具 — 依 CSV 剪輯 MP4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV 欄位（必要：Start_Sec, End_Sec）：
  Start_Frame, End_Frame, Start_Sec, End_Sec, Duration_Sec

範例：
  分段輸出：
    python video_cutter.py -v test2.mp4 -c r.csv -m separate -o ./output_clips

  合併輸出：
    python video_cutter.py -v input.mp4 -c clips.csv -m merge -o merged.mp4

    反向合併輸出（刪除 CSV 片段）：
        python video_cutter.py -v test2.mp4 -c r.csv -m inverse-merge -o inverse_merged.mp4
"""
    )
    parser.add_argument("-v", "--video",  required=True, help="MP4 影片路徑")
    parser.add_argument("-c", "--csv",    required=True, help="CSV 檔案路徑")
    parser.add_argument("-m", "--mode",   required=True,
                        choices=["separate", "merge", "inverse-merge"],
                        help="輸出模式：separate（分段）、merge（合併）或 inverse-merge（刪除 CSV 片段後合併）")
    parser.add_argument("-o", "--output", required=True,
                        help="separate 模式：輸出資料夾；merge 模式：輸出 MP4 路徑")

    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"[錯誤] 影片不存在：{args.video}")
        sys.exit(1)
    if not os.path.isfile(args.csv):
        print(f"[錯誤] CSV 不存在：{args.csv}")
        sys.exit(1)

    try:
        segments = parse_csv(args.csv)
    except Exception as e:
        print(f"[錯誤] 讀取 CSV 失敗：{e}")
        sys.exit(1)

    if not segments:
        print("[錯誤] CSV 中沒有有效片段。")
        sys.exit(1)

    print(f"  讀取到 {len(segments)} 個有效片段。")

    if args.mode == "separate":
        mode_separate(args.video, segments, args.output)
    elif args.mode == "merge":
        mode_merge(args.video, segments, args.output)
    else:
        mode_inverse_merge(args.video, segments, args.output)


# ──────────────────────────────────────────────
#  主程式
# ──────────────────────────────────────────────

def main():
    if not check_ffmpeg():
        sys.exit(1)

    if len(sys.argv) > 1:
        cli_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
