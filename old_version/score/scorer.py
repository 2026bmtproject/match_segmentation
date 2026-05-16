"""Score reading: query each frame for the scoreboard value.

To swap in a different algorithm (e.g. local OCR, another LLM), subclass
ScoreReader and implement ``query_batch``.

Example:
    class TesseractReader(ScoreReader):
        async def query_batch(self, items):
            ...
"""
from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod

import cv2
from google import genai
from google.genai import types

from config import MAX_RETRIES, RETRY_FRAME_POSITIONS
from frame_extractor import extract_frame_at_position, extract_frame_by_number
from models import BBox, SegmentSource
from prompts import SCORE_PROMPT_BATCH, SCORE_PROMPT_SINGLE


# ── confidence ranking helper ─────────────────────────────────────────────────

_CONF_RANK = {"high": 2, "medium": 1, "low": 0}


def best_result(candidates: list[dict]) -> dict:
    return max(candidates, key=lambda r: _CONF_RANK.get(r.get("confidence", "low"), 0))


# ── abstract interface ────────────────────────────────────────────────────────

class ScoreReader(ABC):
    @abstractmethod
    async def query_batch(self, items: list[tuple[str, bytes]]) -> list[dict]:
        """Return one result dict per item (same order).

        Each dict must contain at least: clip, score_a, score_b, confidence, note.
        """

    async def query_single(self, label: str, jpeg_bytes: bytes) -> dict:
        results = await self.query_batch([(label, jpeg_bytes)])
        return results[0]


# ── retry wrapper (algorithm-agnostic) ───────────────────────────────────────

async def retry_uncertain(
    scorer: ScoreReader,
    result: dict,
    source: SegmentSource,
    crop_boxes: list[BBox] | None = None,
) -> dict:
    """If confidence is medium/low, sample extra frames and pick the best result."""
    if result.get("confidence") == "high":
        return result

    extra_items: list[tuple[str, bytes]] = []
    for pos in RETRY_FRAME_POSITIONS:
        if source.mode == "csv" and source.video_path and source.start_frame is not None:
            data = extract_frame_at_position(
                source.video_path,
                source.start_frame,
                source.end_frame or source.start_frame,
                pos,
                crop_boxes,
            )
        elif source.mode == "clip" and source.clip_path:
            cap   = cv2.VideoCapture(str(source.clip_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            data = (
                extract_frame_by_number(source.clip_path, int(total * pos), crop_boxes)
                if total > 0
                else None
            )
        else:
            data = None

        if data:
            extra_items.append((f"{source.label}_pos{int(pos * 100)}", data))

    if not extra_items:
        return result

    print(f"  [重試] {source.label}: confidence={result.get('confidence')}，"
          f"補充 {len(extra_items)} 幀重查...")
    retry_results = await scorer.query_batch(extra_items)
    for r in retry_results:
        r["clip"] = source.label
    return best_result([result] + retry_results)


# ── Gemini implementation ─────────────────────────────────────────────────────

def _parse_retry_delay(err_str: str, fallback: float = 120.0) -> float:
    m = re.search(r'retryDelay["\s:\']+(\d+)', err_str)
    if not m:
        m = re.search(r'retry in (\d+)', err_str, re.IGNORECASE)
    return float(m.group(1)) + 5.0 if m else fallback


def _strip_markdown(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return text


class GeminiScoreReader(ScoreReader):
    def __init__(self, client: genai.Client, sem: asyncio.Semaphore, model: str):
        self.client = client
        self.sem    = sem
        self.model  = model

    async def query_batch(self, items: list[tuple[str, bytes]]) -> list[dict]:
        """Pack all items into one generate_content call; fall back to per-image on parse failure."""
        async with self.sem:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    return await self._call(items)
                except Exception as e:
                    err_str = str(e)
                    is_rate = "429" in err_str or "503" in err_str

                    if is_rate and "limit: 0" in err_str:
                        raise RuntimeError(
                            f"API 配額上限為 0（免費方案不支援 {self.model}）\n"
                            "  請改用：python score_labeler.py --model gemini-1.5-flash"
                        ) from e

                    if not is_rate or attempt >= MAX_RETRIES:
                        if len(items) > 1:
                            print(f"  [警告] 批次解析失敗，改為逐一查詢: {e}")
                            results = []
                            for single in items:
                                r = await self.query_batch([single])
                                results.extend(r)
                            return results
                        raise

                    fallback = 30.0 if "503" in err_str else 120.0
                    delay    = _parse_retry_delay(err_str, fallback)
                    code     = "429" if "429" in err_str else "503"
                    print(f"\n  [{code}] 速率限制，等待 {delay:.0f}s 後重試"
                          f"（{attempt + 1}/{MAX_RETRIES}）...")
                    await asyncio.sleep(delay)

        raise RuntimeError(f"超過最大重試次數 ({MAX_RETRIES})")

    async def _call(self, items: list[tuple[str, bytes]]) -> list[dict]:
        if len(items) == 1:
            return await self._call_single(*items[0])
        return await self._call_batch(items)

    async def _call_single(self, label: str, jpeg_bytes: bytes) -> list[dict]:
        print(f"  [API] 發送 {label} ...")
        image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        response   = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[image_part, SCORE_PROMPT_SINGLE],
            config=types.GenerateContentConfig(
                **{"response_mime_type": "application/json", "temperature": 0}
            ),
        )
        await asyncio.sleep(1)
        parsed = json.loads(_strip_markdown(response.text or ""))
        parsed["clip"] = label
        return [parsed]

    async def _call_batch(self, items: list[tuple[str, bytes]]) -> list[dict]:
        labels = [lbl for lbl, _ in items]
        print(f"  [API] 批次發送 {len(items)} 張（{labels[0]} … {labels[-1]}）...")
        contents: list = []
        for i, (label, jpeg_bytes) in enumerate(items, start=1):
            contents.append(f"[圖片 {i}]")
            contents.append(types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"))
        contents.append(SCORE_PROMPT_BATCH.format(n=len(items)))

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                **{"response_mime_type": "application/json", "temperature": 0}
            ),
        )
        await asyncio.sleep(1)
        parsed_list = json.loads(_strip_markdown(response.text or ""))
        if not isinstance(parsed_list, list):
            raise ValueError("回應不是 JSON 陣列")

        index_map: dict[int, dict] = {
            int(r["image_index"]): r
            for r in parsed_list
            if isinstance(r, dict) and r.get("image_index") is not None
        }
        results = []
        for i, (label, _) in enumerate(items, start=1):
            entry = index_map.get(i) or index_map.get(i - 1)
            if entry is None and i - 1 < len(parsed_list):
                entry = parsed_list[i - 1]
            if entry is None:
                entry = {"score_a": None, "score_b": None, "confidence": "low",
                         "note": "batch response missing"}
            d = dict(entry)
            d["clip"] = label
            d.pop("image_index", None)
            results.append(d)
        return results
