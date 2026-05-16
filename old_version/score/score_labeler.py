"""
score_labeler.py — CLI entry point

使用方式:
    python score_labeler.py --video test.mp4 --csv r.csv
    python score_labeler.py --video test.mp4 --csv r.csv --detect-samples 5
    python score_labeler.py --video test.mp4 --csv r.csv --skip-scoreboard-detection
    python score_labeler.py --video test.mp4 --csv r.csv --crops-only
    python score_labeler.py --model gemini-2.5-flash --batch-size 8 --save-frames frames_preview
    --skip-scoreboard-detection 和 --crops-only 兩個選項會跳過記分板偵測，直接用全幀擷取並（如果有指定 --save-frames）儲存壓縮後的全幀圖。前者會送出 API 請求並嘗試辨識比分，後者則僅儲存圖片不呼叫 API。

請用 $env:GEMINI_API_KEY = "你的_api_key" 設定 key
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

try:
    from google import genai
except ImportError:
    print("請安裝 google-genai: pip install google-genai")
    sys.exit(1)

from config import (
    API_KEY_ENV, BATCH_SIZE, CLIPS_DIR, COMPRESS_H, COMPRESS_W,
    JPEG_QUALITY, MAX_CONCURRENT,
    MIN_BINARY_INTERVAL_SEC, OUTPUT_JSON, PRICE_INPUT_PER_M,
    PRICE_OUTPUT_PER_M, RETRY_FRAME_POSITIONS, TOKENS_IMAGE,
    TOKENS_OUTPUT, TOKENS_PROMPT, MODEL,
)
from csv_parser import parse_csv_segments
from detector import GeminiDetector, detect_sample_count
from frame_extractor import extract_frame_by_number, extract_middle_frame_bytes, extract_raw_frame
from gap_filler import fill_gaps
from grouper import group_by_score
from models import BBox, SegmentSource
from scorer import GeminiScoreReader, retry_uncertain


async def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini 羽球比賽記分板辨識工具")
    parser.add_argument("--model",       default=MODEL)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--save-frames", metavar="DIR", default=None)
    parser.add_argument("--video",       metavar="PATH", default=None,
                        help="來源影片路徑（需與 --csv 搭配）")
    parser.add_argument("--csv",         metavar="PATH", default=None,
                        help="CSV 分段檔案路徑（需與 --video 搭配）")
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE,
                        help=f"每次 API 請求的圖片數（預設 {BATCH_SIZE}）")
    parser.add_argument("--min-binary-interval", type=float, default=MIN_BINARY_INTERVAL_SEC,
                        help=f"二分搜尋最小間距秒數（預設 {MIN_BINARY_INTERVAL_SEC}s）")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"最大並行 API 請求數（預設 {MAX_CONCURRENT}）")
    parser.add_argument("--detect-samples", type=int, default=-1, metavar="N",
                        help="偵測記分板用的取樣幀數（-1=自動按片段數%計算，0=停用，N=指定張數）")
    parser.add_argument("--skip-scoreboard-detection", action="store_true",
                        help="直接跳過記分板偵測，改用全幀擷取")
    parser.add_argument("--crops-only", action="store_true",
                        help="偵測記分板後僅儲存裁切圖，不呼叫評分 API")
    args  = parser.parse_args()
    model = args.model

    if bool(args.video) != bool(args.csv):
        parser.error("--video 和 --csv 必須同時提供或同時省略")
    if args.crops_only and not args.save_frames:
        args.save_frames = "crops"

    csv_mode = args.video is not None

    # ── list models ───────────────────────────────────────────────────────────
    if args.list_models:
        api_key = os.environ.get(API_KEY_ENV) or input("請輸入 Gemini API Key: ").strip()
        client  = genai.Client(api_key=api_key)
        print("支援 generateContent 的模型：")
        for m in client.models.list():
            if "generateContent" in (m.supported_actions or []):
                print(f"  {m.name}")
        sys.exit(0)

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"未設定環境變數 {API_KEY_ENV}")
        api_key = input("請輸入 Gemini API Key: ").strip()
        if not api_key:
            sys.exit(1)

    # ── scan inputs ───────────────────────────────────────────────────────────
    csv_segments: list[dict] = []
    clips: list[Path] = []

    if csv_mode:
        print(f"讀取 CSV: {args.csv}")
        csv_segments = parse_csv_segments(Path(args.csv))
        n = len(csv_segments)
        print(f"共 {n} 個 CSV 分段")
    else:
        clips = sorted(CLIPS_DIR.glob("*.mp4"))
        if not clips:
            print(f"在 {CLIPS_DIR}/ 找不到 .mp4 檔案")
            sys.exit(1)
        n = len(clips)
        print(f"找到 {n} 個片段")

    if n == 0:
        print("沒有可處理的影片，結束。")
        sys.exit(1)

    # ── cost estimate ─────────────────────────────────────────────────────────
    batch_size   = max(1, args.batch_size)
    n_batches    = math.ceil(n / batch_size)
    est_retry    = int(n * 0.2 * len(RETRY_FRAME_POSITIONS))
    board_detection_enabled = not args.skip_scoreboard_detection and args.detect_samples != 0
    if not board_detection_enabled:
        detect_calls = 0
    elif args.detect_samples > 0:
        detect_calls = min(args.detect_samples, n)
    else:  # -1 → auto
        detect_calls = min(detect_sample_count(n), n)

    if args.crops_only:
        total_in  = detect_calls * TOKENS_IMAGE + detect_calls * TOKENS_PROMPT
        total_out = detect_calls * TOKENS_OUTPUT
    else:
        total_in  = ((n + est_retry + detect_calls) * TOKENS_IMAGE
                     + (n_batches + detect_calls) * TOKENS_PROMPT)
        total_out = n * TOKENS_OUTPUT

    cost_usd = (total_in  / 1_000_000 * PRICE_INPUT_PER_M
                + total_out / 1_000_000 * PRICE_OUTPUT_PER_M)

    print()
    print("=" * 62)
    print(f"  Gemini Score Labeler — {model}")
    print("=" * 62)
    mode_label = "CSV+影片" if csv_mode else "片段資料夾"
    if args.skip_scoreboard_detection:
        mode_label += "  [skip-board-detection]"
    if args.crops_only:
        mode_label += "  [crops-only]"
    print(f"  模式:               {mode_label}")
    print(f"  圖片數:             {n}")
    if not args.crops_only:
        print(f"  批次大小:           {batch_size}  → {n_batches} 次 API 請求")
    print(f"  壓縮尺寸:           {COMPRESS_W}×{COMPRESS_H} px，JPEG Q{JPEG_QUALITY}")
    if args.skip_scoreboard_detection:
        print("  記分板偵測:         已停用（直接擷取全幀）")
    elif detect_calls:
        auto_note = "（自動）" if args.detect_samples == -1 else ""
        print(f"  記分板偵測樣本:     {detect_calls} 張{auto_note}")
    if args.crops_only:
        print(f"  輸出目錄:           {args.save_frames}/")
        print(f"  估算 tokens:       ~{total_in:,} in / ~{total_out:,} out（僅偵測）")
    else:
        detect_note = f"、{detect_calls} 張偵測" if detect_calls else ""
        print(f"  估算輸入 tokens:   ~{total_in:,}（含預估 {est_retry} 張補充幀{detect_note}）")
        print(f"  估算輸出 tokens:   ~{total_out:,}")
    print(f"  預估費用:          ~${cost_usd:.5f} USD")
    if not args.crops_only:
        print(f"  並行請求上限:       {args.max_concurrent}")
    if csv_mode and not args.crops_only:
        print(f"  二分搜尋最小間距:   {args.min_binary_interval}s")
    print()

    if args.crops_only:
        confirm = input(f"將進行 {detect_calls} 次偵測後儲存裁切圖並結束，確認？(y/N): ").strip().lower()
    else:
        confirm = input("確認送出 API 請求？(y/N): ").strip().lower()
    if confirm != "y":
        print("已取消。")
        sys.exit(0)

    # ── build API objects ─────────────────────────────────────────────────────
    client   = genai.Client(api_key=api_key)
    sem      = asyncio.Semaphore(args.max_concurrent)
    detector = GeminiDetector(client, sem, model)
    scorer   = GeminiScoreReader(client, sem, model)

    # ── scoreboard detection (optional) ──────────────────────────────────────
    crop_boxes: list[BBox] = []
    if detect_calls > 0:
        step = max(1, n // detect_calls)
        if csv_mode:
            sample_frames_raw: list[tuple[str, bytes]] = []
            for seg in csv_segments[::step][:detect_calls]:
                sf, ef = seg["start_frame"], seg["end_frame"]
                mid    = (sf + ef) // 2 if ef > sf else sf
                data   = extract_raw_frame(Path(args.video), mid)
                if data:
                    sample_frames_raw.append((f"frame_{sf}_{ef}", data))
        else:
            import cv2
            sample_frames_raw = []
            for clip in clips[::step][:detect_calls]:
                cap     = cv2.VideoCapture(str(clip))
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
                cap.release()
                data = extract_raw_frame(clip, total_f // 2) if total_f > 0 else None
                if data:
                    sample_frames_raw.append((clip.name, data))

        if sample_frames_raw:
            crop_boxes = await detector.detect(sample_frames_raw)
            if crop_boxes:
                label = "兩個記分板合成" if len(crop_boxes) > 1 else "記分板裁切"
                print(f"\n{label}已啟用（{len(crop_boxes)} 個區域）。")

    # ── extract frames ────────────────────────────────────────────────────────
    crop_note = ""
    if crop_boxes:
        crop_note = f"（{'兩記分板合成' if len(crop_boxes) > 1 else '記分板裁切'}）"

    frames:  dict[str, bytes]         = {}
    sources: dict[str, SegmentSource] = {}

    if csv_mode:
        print(f"\n擷取 {n} 個片段的幀 {crop_note}...")
        for seg in csv_segments:
            sf, ef    = seg["start_frame"], seg["end_frame"]
            mid_frame = (sf + ef) // 2 if ef > sf else sf
            label     = f"frame_{sf}_{ef}"
            data      = extract_frame_by_number(Path(args.video), mid_frame, crop_boxes)
            if data is None:
                print(f"  [跳過] {label}: 無法讀取")
                continue
            frames[label]  = data
            sources[label] = SegmentSource(
                label=label, mode="csv",
                clip_path=None, video_path=Path(args.video),
                start_frame=sf, end_frame=ef,
                start_sec=seg["start_sec"], end_sec=seg["end_sec"],
            )
    else:
        print(f"\n擷取 {n} 個片段的中間幀 {crop_note}...")
        for clip in clips:
            data = extract_middle_frame_bytes(clip, crop_boxes)
            if data is None:
                print(f"  [跳過] {clip.name}: 無法讀取")
                continue
            frames[clip.name]  = data
            sources[clip.name] = SegmentSource(
                label=clip.name, mode="clip",
                clip_path=clip, video_path=None,
                start_frame=None, end_frame=None,
            )

    if not frames:
        print("沒有可處理的影片，結束。")
        sys.exit(1)

    # ── save frame previews ───────────────────────────────────────────────────
    if args.save_frames:
        frames_dir = Path(args.save_frames)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for name, data in frames.items():
            (frames_dir / (Path(name).stem + ".jpg")).write_bytes(data)
        suffix = f"（{crop_note.strip('（）')}）" if crop_boxes else ""
        print(f"\n壓縮幀已儲存至 {frames_dir}/ （共 {len(frames)} 張）{suffix}")

    if args.crops_only:
        board_str = "兩記分板合成" if len(crop_boxes) > 1 else "記分板裁切" if crop_boxes else "全幀"
        print(f"\n[crops-only] {board_str}圖已輸出 {len(frames)} 張至 {args.save_frames}/，結束程式。")
        sys.exit(0)

    # ── batch score query ─────────────────────────────────────────────────────
    frame_items = list(frames.items())
    batches     = [frame_items[i:i + batch_size] for i in range(0, len(frame_items), batch_size)]

    print(f"\n送出 {n} 張圖（共 {len(batches)} 批，每批最多 {batch_size} 張）...")
    batch_tasks    = [scorer.query_batch(batch) for batch in batches]
    gather_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

    raw_results: list[dict] = []
    for batch_idx, batch_result in enumerate(gather_results):
        if isinstance(batch_result, Exception):
            print(f"  [錯誤] 批次 {batch_idx + 1}: {batch_result}")
            for label, _ in batches[batch_idx]:
                raw_results.append({
                    "clip": label, "score_a": None, "score_b": None,
                    "confidence": "low", "note": f"API 錯誤: {batch_result}",
                })
        elif isinstance(batch_result, list):
            for r in batch_result:
                a, b, conf = r.get("score_a"), r.get("score_b"), r.get("confidence", "?")
                print(f"  {r['clip']}: A={a}  B={b}  [{conf}]")
                raw_results.append(r)

    # ── retry uncertain ───────────────────────────────────────────────────────
    uncertain = [
        r for r in raw_results
        if r.get("confidence") in ("medium", "low") and r["clip"] in sources
    ]
    if uncertain:
        print(f"\n對 {len(uncertain)} 個不確定結果進行補充幀重查...")
        retry_tasks = [
            retry_uncertain(scorer, r, sources[r["clip"]], crop_boxes or None)
            for r in uncertain
        ]
        retry_outcomes = await asyncio.gather(*retry_tasks, return_exceptions=True)
        result_map = {r["clip"]: r for r in raw_results}
        for orig_r, outcome in zip(uncertain, retry_outcomes):
            if isinstance(outcome, Exception):
                print(f"  [重試錯誤] {orig_r['clip']}: {outcome}")
            elif isinstance(outcome, dict):
                if outcome.get("confidence") != orig_r.get("confidence"):
                    print(f"  [重試改善] {orig_r['clip']}: "
                          f"{orig_r.get('confidence')} → {outcome.get('confidence')} "
                          f"A={outcome.get('score_a')} B={outcome.get('score_b')}")
                result_map[orig_r["clip"]] = outcome
        raw_results = list(result_map.values())

    # ── CSV mode: annotate metadata + gap fill ────────────────────────────────
    if csv_mode:
        seg_by_label = {f"frame_{s['start_frame']}_{s['end_frame']}": s for s in csv_segments}
        for r in raw_results:
            seg = seg_by_label.get(r["clip"])
            if seg:
                r.setdefault("start_frame",    seg["start_frame"])
                r.setdefault("end_frame",      seg["end_frame"])
                r.setdefault("start_sec",      seg["start_sec"])
                r.setdefault("end_sec",        seg["end_sec"])
            r.setdefault("gap_fill",        False)
            r.setdefault("source_time_sec", None)

        print("\n執行二分搜尋補漏...")
        raw_results = await fill_gaps(
            scorer=scorer,
            raw_results=raw_results,
            csv_segments=csv_segments,
            video_path=Path(args.video),
            min_interval=args.min_binary_interval,
            crop_boxes=crop_boxes or None,
        )

    # ── group & output ────────────────────────────────────────────────────────
    print("\n依比分分組並檢查異常...")
    groups           = group_by_score(raw_results)
    gap_fill_results = [r for r in raw_results if r.get("gap_fill")]

    output = {
        "model":        model,
        "mode":         "csv" if csv_mode else "clips",
        "source_video": args.video if csv_mode else None,
        "source_csv":   args.csv   if csv_mode else None,
        "clips_dir":    str(CLIPS_DIR) if not csv_mode else None,
        "compress_size": f"{COMPRESS_W}x{COMPRESS_H}",
        "scoreboard_bbox":  crop_boxes[0].to_dict() if len(crop_boxes) == 1 else None,
        "scoreboard_bboxes": [b.to_dict() for b in crop_boxes] if crop_boxes else None,
        "batch_size":   batch_size,
        "total_clips":  n,
        "raw_results":  sorted(raw_results, key=lambda r: r["clip"]),
        "gap_fill_results": gap_fill_results,
        "binary_search_summary": {
            "gaps_found":      sum(1 for r in gap_fill_results if r.get("score_a") is not None),
            "missing_rallies": sum(1 for r in gap_fill_results if r.get("score_a") is None),
        },
        "groups": groups,
    }
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果已儲存至 {OUTPUT_JSON}")

    # ── summary ───────────────────────────────────────────────────────────────
    review_count = sum(1 for g in groups if g["needs_review"])
    print()
    print("=" * 62)
    print(f"  分組摘要（共 {len(groups)} 組，{review_count} 組需確認）")
    print("=" * 62)
    for g in groups:
        flag      = "⚠ " if g["needs_review"] else "  "
        score_str = (f"A:{g['score_a']:2d}  B:{g['score_b']:2d}"
                     if g["score_a"] is not None else "無法辨識      ")
        reason    = f"  ← {g['review_reason']}" if g["review_reason"] else ""
        print(f"  {flag}{score_str} | {len(g['clips'])} 片段 | {', '.join(g['clips'])}{reason}")

    if crop_boxes:
        print()
        for i, b in enumerate(crop_boxes, 1):
            lbl = f"記分板 {i}" if len(crop_boxes) > 1 else "記分板裁切"
            print(f"  {lbl}: x=[{b.x1:.3f}, {b.x2:.3f}]  y=[{b.y1:.3f}, {b.y2:.3f}]")

    if csv_mode and gap_fill_results:
        gaps    = output["binary_search_summary"]["gaps_found"]
        missing = output["binary_search_summary"]["missing_rallies"]
        print()
        print(f"  二分搜尋：找到中間幀 {gaps} 個，標記缺漏回合 {missing} 個")


if __name__ == "__main__":
    asyncio.run(main())
