"""
score_labeler.py

讀取 output_clips/ 內所有影片，擷取中間幀後呼叫 Gemini 2.0 Flash
辨識記分板分數，依比分將片段分組並標記異常，輸出 JSON 報告。

使用方式:
    python score_labeler.py
    GEMINI_API_KEY=xxx python score_labeler.py
    python score_labeler.py --model gemini-2.5-flash  # 免費可用
    python score_labeler.py --save-frames frames_preview  # 同時儲存壓縮後的幀供人工確認

請用$env:GEMINI_API_KEY = "你的_api_key"設定key，不然會出bug
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("請安裝 google-genai: pip install google-genai")
    sys.exit(1)

# ── 設定 ──────────────────────────────────────────────────────────────────────

CLIPS_DIR    = Path("output_clips")
OUTPUT_JSON  = Path("score_labels.json")
API_KEY_ENV  = "GEMINI_API_KEY"
MODEL        = "gemini-2.5-flash"

# 壓縮尺寸：640×360 → 3×2 tiles → 1548 tokens/圖（比原始 1280×720 少 ~75%）
COMPRESS_W     = 640
COMPRESS_H     = 360
JPEG_QUALITY   = 85
MAX_CONCURRENT = 1     # 同時並行的 API 請求數
MAX_RETRIES    = 3     # 429 限流時的最大重試次數

# Gemini 2.5 Flash 定價（USD / 1M tokens）
PRICE_INPUT_PER_M  = 0.075
PRICE_OUTPUT_PER_M = 0.30

# Token 估算常數
# 640×360: ceil(640/256)=3, ceil(360/256)=2 → 6 tiles × 258 = 1548 tokens
TOKENS_IMAGE   = 1548
TOKENS_PROMPT  = 150   # 系統提示詞
TOKENS_OUTPUT  = 80    # JSON 輸出

SCORE_PROMPT = """\
這是一場羽球比賽的截圖，請找出畫面中記分板上的比分。

請只回傳以下 JSON 格式，不要任何其他文字或 markdown：
{
  "score_a": <左方或上方隊伍的整數分數>,
  "score_b": <右方或下方隊伍的整數分數>,
  "confidence": "<high | medium | low>",
  "note": "<備註，若無請填空字串>"
}

若完全看不到記分板，score_a 和 score_b 請填 null，confidence 填 "low"。\
"""

# ── 工具函式 ──────────────────────────────────────────────────────────────────

def _parse_retry_delay(err_str: str, fallback: float = 120.0) -> float:
    """從 429 錯誤訊息中解析建議等待秒數。"""
    m = re.search(r'retryDelay["\s:\']+(\d+)', err_str)
    if not m:
        m = re.search(r'retry in (\d+)', err_str, re.IGNORECASE)
    return float(m.group(1)) + 5.0 if m else fallback


def extract_middle_frame_bytes(video_path: Path) -> bytes | None:
    """擷取影片中間幀，壓縮後回傳 JPEG bytes。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    frame = cv2.resize(frame, (COMPRESS_W, COMPRESS_H), interpolation=cv2.INTER_AREA)
    ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return bytes(buf) if ok2 else None


async def query_score(
    client: genai.Client,
    sem: asyncio.Semaphore,
    clip_name: str,
    jpeg_bytes: bytes,
    model: str,
) -> dict:
    """向 Gemini 查詢一張圖的比分，429 時自動退避重試，回傳含 'clip' 鍵的 dict。"""
    async with sem:
        for attempt in range(MAX_RETRIES + 1):
            try:
                print(f"  [API] 發送 {clip_name} 請求...")
                image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[image_part, SCORE_PROMPT],
                    config=types.GenerateContentConfig(  # type: ignore[call-arg]
                        **{"response_mime_type": "application/json", "temperature": 0}
                    ),
                )
                print(f"  [API] {clip_name} 請求完成，等待 1 秒...")
                await asyncio.sleep(1)  # 每次調用後等待 1 秒
                break  # 成功則跳出重試迴圈

            except Exception as e:
                err_str = str(e)
                is_429 = "429" in err_str

                # limit: 0 → 免費方案根本沒有此模型的配額，重試無效
                if is_429 and "limit: 0" in err_str:
                    raise RuntimeError(
                        f"API 配額上限為 0（免費方案不支援 {model}）\n"
                        "  請改用：python score_labeler.py --model gemini-1.5-flash\n"
                        "  或前往 https://aistudio.google.com 啟用帳單"
                    ) from e

                if not is_429 or attempt >= MAX_RETRIES:
                    raise

                delay = _parse_retry_delay(err_str)
                print(f"\n  [429] {clip_name} 速率限制，等待 {delay:.0f}s 後重試"
                      f"（{attempt + 1}/{MAX_RETRIES}）...")
                await asyncio.sleep(delay)
        else:
            raise RuntimeError(f"{clip_name}: 超過最大重試次數 ({MAX_RETRIES})")

    text = (response.text or "").strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

    parsed: dict = json.loads(text)
    parsed["clip"] = clip_name
    return parsed


def group_by_score(results: list[dict]) -> list[dict]:
    """
    依 (score_a, score_b) 分組，依總分排序，
    並對不符合羽球計分邏輯的轉換標記 needs_review。
    """
    valid = [r for r in results if r.get("score_a") is not None and r.get("score_b") is not None]
    unreadable = [r for r in results if r.get("score_a") is None or r.get("score_b") is None]

    # 聚合相同分數的片段
    score_map: dict[tuple[int, int], list[str]] = {}
    for r in valid:
        key = (int(r["score_a"]), int(r["score_b"]))
        score_map.setdefault(key, []).append(r["clip"])

    # 依總分（再依 A 分）升冪排序
    sorted_scores = sorted(score_map.items(), key=lambda x: (x[0][0] + x[0][1], x[0][0]))

    groups: list[dict] = []
    prev: tuple[int, int] | None = None

    for (a, b), clips in sorted_scores:
        needs_review = False
        reason = ""

        if prev is not None:
            pa, pb = prev
            da, db = a - pa, b - pb

            if da < 0 or db < 0:
                needs_review = True
                reason = f"分數倒退: ({pa},{pb}) → ({a},{b})"
            elif da > 0 and db > 0:
                needs_review = True
                reason = f"雙方同時加分 (不合理): ({pa},{pb}) → ({a},{b})"
            elif da + db > 1:
                needs_review = True
                reason = f"跳 {da+db} 分 (可能有缺漏片段): ({pa},{pb}) → ({a},{b})"

        groups.append({
            "score_a": a,
            "score_b": b,
            "clips": sorted(clips),
            "needs_review": needs_review,
            "review_reason": reason,
        })
        prev = (a, b)

    if unreadable:
        groups.append({
            "score_a": None,
            "score_b": None,
            "clips": sorted(r["clip"] for r in unreadable),
            "needs_review": True,
            "review_reason": "無法辨識分數",
        })

    return groups


# ── 主程式 ────────────────────────────────────────────────────────────────────

async def main() -> None:
    # ── 參數解析 ──────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Gemini 羽球比賽記分板辨識工具")
    parser.add_argument(
        "--model", default=MODEL,
        help=f"Gemini 模型名稱（預設: {MODEL}）",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="列出此 API Key 可用的模型後結束",
    )
    parser.add_argument(
        "--save-frames", metavar="DIR", default=None,
        help="將壓縮後的中間幀儲存至指定資料夾供人工確認（選填）",
    )
    args = parser.parse_args()
    model = args.model

    # ── 列出可用模型 ──────────────────────────────────────
    if args.list_models:
        api_key = os.environ.get(API_KEY_ENV)
        if not api_key:
            api_key = input("請輸入 Gemini API Key: ").strip()
        client = genai.Client(api_key=api_key)
        print("支援 generateContent 的模型：")
        for m in client.models.list():
            if "generateContent" in (m.supported_actions or []):
                print(f"  {m.name}")
        sys.exit(0)

    # ── API Key ───────────────────────────────────────────
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"未設定環境變數 {API_KEY_ENV}")
        api_key = input("請輸入 Gemini API Key: ").strip()
        if not api_key:
            sys.exit(1)

    # ── 掃描影片 ──────────────────────────────────────────
    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    if not clips:
        print(f"在 {CLIPS_DIR}/ 找不到 .mp4 檔案")
        sys.exit(1)

    n = len(clips)

    # ── 費用估算 ──────────────────────────────────────────
    total_in  = n * (TOKENS_IMAGE + TOKENS_PROMPT)
    total_out = n * TOKENS_OUTPUT
    cost_usd  = (total_in / 1_000_000 * PRICE_INPUT_PER_M
                 + total_out / 1_000_000 * PRICE_OUTPUT_PER_M)

    print("=" * 62)
    print(f"  Gemini Score Labeler — {model}")
    print("=" * 62)
    print(f"  影片數:             {n}")
    print(f"  壓縮尺寸:           {COMPRESS_W}×{COMPRESS_H} px，JPEG Q{JPEG_QUALITY}")
    print(f"  原始解析度:         1280×720 px（壓縮節省約 75% tokens）")
    print(f"  估算輸入 tokens:   ~{total_in:,}")
    print(f"    └ 圖片 {TOKENS_IMAGE}/張 + 提示 {TOKENS_PROMPT}/張 × {n} 張")
    print(f"  估算輸出 tokens:   ~{total_out:,}  ({TOKENS_OUTPUT}/張)")
    print(f"  預估費用:          ~${cost_usd:.5f} USD")
    print(f"  並行請求上限:       {MAX_CONCURRENT}")
    print()

    confirm = input("確認送出 API 請求？(y/N): ").strip().lower()
    if confirm != "y":
        print("已取消。")
        sys.exit(0)

    # ── 擷取中間幀 ────────────────────────────────────────
    print("\n擷取並壓縮中間幀...")
    frames: dict[str, bytes] = {}
    for clip in clips:
        data = extract_middle_frame_bytes(clip)
        if data is None:
            print(f"  [跳過] {clip.name}：無法讀取")
            continue
        frames[clip.name] = data
        print(f"  {clip.name}: {len(data) / 1024:.1f} KB")

    if not frames:
        print("沒有可處理的影片，結束。")
        sys.exit(1)

    # ── 儲存壓縮幀（選填）────────────────────────────────
    if args.save_frames:
        frames_dir = Path(args.save_frames)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for name, data in frames.items():
            out_path = frames_dir / (Path(name).stem + ".jpg")
            out_path.write_bytes(data)
        print(f"\n壓縮幀已儲存至 {frames_dir}/ （共 {len(frames)} 張）")

    # ── 呼叫 Gemini ───────────────────────────────────────
    print(f"\n送出 {len(frames)} 個請求（最多 {MAX_CONCURRENT} 個並行）...")
    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        query_score(client, sem, name, data, model)
        for name, data in frames.items()
    ]
    gather_results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── 處理回傳 ──────────────────────────────────────────
    raw_results: list[dict] = []
    for clip_name, result in zip(frames.keys(), gather_results):
        if isinstance(result, Exception):
            print(f"  [錯誤] {clip_name}: {result}")
            raw_results.append({
                "clip": clip_name,
                "score_a": None,
                "score_b": None,
                "confidence": "low",
                "note": f"API 錯誤: {result}",
            })
        elif isinstance(result, dict):
            a    = result.get("score_a")
            b    = result.get("score_b")
            conf = result.get("confidence", "?")
            print(f"  {clip_name}: A={a}  B={b}  [{conf}]")
            raw_results.append(result)

    # ── 分組 ─────────────────────────────────────────────
    print("\n依比分分組並檢查異常...")
    groups = group_by_score(raw_results)

    # ── 儲存 JSON ─────────────────────────────────────────
    output = {
        "model": MODEL,
        "clips_dir": str(CLIPS_DIR),
        "compress_size": f"{COMPRESS_W}x{COMPRESS_H}",
        "total_clips": n,
        "raw_results": sorted(raw_results, key=lambda r: r["clip"]),
        "groups": groups,
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n結果已儲存至 {OUTPUT_JSON}")

    # ── 摘要 ─────────────────────────────────────────────
    review_count = sum(1 for g in groups if g["needs_review"])
    print()
    print("=" * 62)
    print(f"  分組摘要（共 {len(groups)} 組，{review_count} 組需確認）")
    print("=" * 62)
    for g in groups:
        flag = "⚠ " if g["needs_review"] else "  "
        if g["score_a"] is not None:
            score_str = f"A:{g['score_a']:2d}  B:{g['score_b']:2d}"
        else:
            score_str = "無法辨識      "
        reason_str = f"  ← {g['review_reason']}" if g["review_reason"] else ""
        clip_list  = ", ".join(g["clips"])
        print(f"  {flag}{score_str} | {len(g['clips'])} 片段 | {clip_list}{reason_str}")


if __name__ == "__main__":
    asyncio.run(main())
