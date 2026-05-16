"""
ocr_compare.py — EasyOCR vs PaddleOCR 視覺化比較工具

用法:
    python ocr_compare.py --video test.mp4 --time 400
    python ocr_compare.py --video test.mp4 --time 6:35
    python ocr_compare.py --video test.mp4 --time 0:01:30 --output result.png
    python ocr_compare.py --video test.mp4 --time 90 --no-gpu
    python ocr_compare.py --video test.mp4 --time 90 --lang en

依賴:
    pip install easyocr
    pip install paddlepaddle paddleocr
    pip install opencv-python pillow
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── time / font helpers ───────────────────────────────────────────────────────

def parse_time(t: str) -> float:
    """接受 '90'、'1:30'、'0:01:30' 格式，回傳秒數。"""
    if ":" in t:
        parts = t.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(t)


def _get_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",       # Microsoft YaHei (支援中文)
        "C:/Windows/Fonts/msjh.ttc",       # Microsoft JhengHei
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── frame extraction ──────────────────────────────────────────────────────────

def extract_frame(video_path: str, time_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  無法開啟影片: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = min(int(time_sec * fps), max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, frame = cap.read()
    cap.release()
    actual_t = frame_num / fps
    print(f"  fps={fps:.2f}  frame={frame_num}/{total}  actual_t={actual_t:.3f}s")
    return frame if ok else None


# ── bounding-box drawing (OpenCV, BGR) ────────────────────────────────────────

_COLOR_EASY   = (50, 210, 80)    # green (BGR)
_COLOR_PADDLE = (40, 50, 220)    # red   (BGR)


def _poly_label(img: np.ndarray, pts: np.ndarray, label: str, color: tuple) -> None:
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    x, y = int(pts[0][0]), int(pts[0][1])
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    y0 = max(y - 4, th + 6)
    cv2.rectangle(img, (x, y0 - th - 4), (x + tw + 4, y0 + 2), color, -1)
    cv2.putText(img, label, (x + 2, y0), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (0, 0, 0), 1, cv2.LINE_AA)


def annotate_easyocr(frame: np.ndarray, results: list) -> np.ndarray:
    img = frame.copy()
    for bbox, text, conf in results:
        pts = np.array(bbox, dtype=np.int32)
        _poly_label(img, pts, f"{conf:.2f}", _COLOR_EASY)
    return img


def annotate_paddle(frame: np.ndarray, results: list) -> np.ndarray:
    img = frame.copy()
    for item in results:
        bbox, (text, conf) = item[0], item[1]
        pts = np.array(bbox, dtype=np.int32)
        _poly_label(img, pts, f"{conf:.2f}", _COLOR_PADDLE)
    return img


# ── PIL text panels (supports Chinese) ────────────────────────────────────────

def _make_text_panel(
    entries: list[tuple[str, float]],
    width: int,
    header: str,
    header_color: tuple,   # RGB
    font_size: int = 17,
) -> np.ndarray:
    """Returns numpy array in RGB."""
    font  = _get_font(font_size)
    small = _get_font(max(font_size - 3, 12))
    lh = font_size + 10
    header_h = font_size + 14
    body_h = max(len(entries), 1) * lh + 16
    h = header_h + body_h

    img  = Image.new("RGB", (width, h), (28, 28, 28))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, header_h], fill=header_color)
    draw.text((10, 7), header, font=font, fill=(255, 255, 255))

    y = header_h + 8
    if not entries:
        draw.text((10, y), "(未偵測到文字)", font=small, fill=(140, 140, 140))
    for text, conf in entries:
        draw.text((10, y), f"[{conf:.3f}]  {text}", font=small, fill=(220, 220, 220))
        y += lh

    return np.array(img)  # RGB


def _pil_label_bar(text: str, w: int, bg_rgb: tuple, font_size: int = 16) -> np.ndarray:
    """Colored header bar — returns BGR numpy array."""
    h = font_size + 22
    bar = Image.new("RGB", (w, h), bg_rgb)
    draw = ImageDraw.Draw(bar)
    draw.text((10, 10), text, font=_get_font(font_size), fill=(255, 255, 255))
    return cv2.cvtColor(np.array(bar), cv2.COLOR_RGB2BGR)


def _bbox_to_jsonable(bbox: list) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in bbox]


# ── composite image ───────────────────────────────────────────────────────────

def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    scale = h / img.shape[0]
    w = max(1, int(img.shape[1] * scale))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _pad_h(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] >= target_h:
        return img
    pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return np.vstack([img, pad])


def build_comparison(
    original: np.ndarray,
    easy_ann: np.ndarray,
    paddle_ann: np.ndarray,
    easy_texts: list[tuple[str, float]],
    paddle_texts: list[tuple[str, float]],
    time_sec: float,
    video_name: str,
    easy_elapsed: float,
    paddle_elapsed: float,
) -> np.ndarray:
    FRAME_H = 480

    o = _resize_h(original,   FRAME_H)
    e = _resize_h(easy_ann,   FRAME_H)
    p = _resize_h(paddle_ann, FRAME_H)
    wo, we, wp = o.shape[1], e.shape[1], p.shape[1]

    # label bars
    orig_bar   = _pil_label_bar("Original Frame", wo, (55, 55, 55))
    easy_bar   = _pil_label_bar(
        f"EasyOCR  [{easy_elapsed:.2f}s]  {len(easy_texts)} 筆", we, (20, 130, 50))
    paddle_bar = _pil_label_bar(
        f"PaddleOCR  [{paddle_elapsed:.2f}s]  {len(paddle_texts)} 筆", wp, (170, 35, 35))

    orig_col   = np.vstack([orig_bar,   o])
    easy_col   = np.vstack([easy_bar,   e])
    paddle_col = np.vstack([paddle_bar, p])

    col_h = max(orig_col.shape[0], easy_col.shape[0], paddle_col.shape[0])
    frame_row = np.hstack([
        _pad_h(orig_col,   col_h),
        _pad_h(easy_col,   col_h),
        _pad_h(paddle_col, col_h),
    ])
    total_w = frame_row.shape[1]

    # text panels (PIL RGB → BGR)
    def rgb_panel_bgr(entries, w, header, hdr_color):
        arr = _make_text_panel(entries, w, header, hdr_color)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    orig_p   = rgb_panel_bgr([], wo,
        "原始影格", (60, 60, 60))
    easy_p   = rgb_panel_bgr(easy_texts, we,
        f"EasyOCR 偵測結果  ({len(easy_texts)} 筆)", (20, 130, 50))
    paddle_p = rgb_panel_bgr(paddle_texts, wp,
        f"PaddleOCR 偵測結果  ({len(paddle_texts)} 筆)", (170, 35, 35))

    panel_h = max(orig_p.shape[0], easy_p.shape[0], paddle_p.shape[0])
    text_row = np.hstack([
        _pad_h(orig_p,   panel_h),
        _pad_h(easy_p,   panel_h),
        _pad_h(paddle_p, panel_h),
    ])

    # title bar
    title_txt = f"OCR Compare  |  {video_name}  |  t = {time_sec:.3f}s"
    title_bar_img = Image.new("RGB", (total_w, 52), (18, 18, 65))
    ImageDraw.Draw(title_bar_img).text(
        (14, 12), title_txt, font=_get_font(20), fill=(255, 218, 80))
    title_bar = cv2.cvtColor(np.array(title_bar_img), cv2.COLOR_RGB2BGR)

    divider = np.full((3, total_w, 3), (90, 90, 90), dtype=np.uint8)

    return np.vstack([title_bar, frame_row, divider, text_row])


# ── OCR runners ───────────────────────────────────────────────────────────────

def run_easyocr(frame: np.ndarray, langs: list[str], use_gpu: bool):
    try:
        import easyocr
    except ImportError:
        print("  ✗ easyocr 未安裝  →  pip install easyocr")
        return [], 0.0

    print(f"  初始化 EasyOCR (langs={langs}, gpu={use_gpu})...")
    reader = easyocr.Reader(langs, gpu=use_gpu)
    t0 = time.perf_counter()
    results = reader.readtext(frame)
    elapsed = time.perf_counter() - t0
    print(f"  完成 ({elapsed:.2f}s) — 偵測到 {len(results)} 筆")
    for _, text, conf in results:
        print(f"    [{conf:.3f}] {text}")
    return results, elapsed


def run_paddleocr(frame: np.ndarray, use_gpu: bool):
    try:
        from paddleocr import PaddleOCR
        import paddle
    except ImportError:
        print("  ✗ paddleocr 未安裝  →  pip install paddlepaddle paddleocr")
        return [], 0.0

    paddle_gpu_available = bool(getattr(paddle, "is_compiled_with_cuda", lambda: False)())
    device = "gpu" if use_gpu and paddle_gpu_available else "cpu"
    if use_gpu and not paddle_gpu_available:
        print("  偵測到目前 PaddlePaddle 未編譯 CUDA，改用 CPU")

    print(f"  初始化 PaddleOCR (device={device})...")
    paddle = PaddleOCR(
        use_textline_orientation=True,
        lang="ch",
        device=device,
        engine="paddle_dynamic",
    )
    t0 = time.perf_counter()
    raw = paddle.predict(frame)
    elapsed = time.perf_counter() - t0

    results: list = []
    if raw:
        block = raw[0]
        for bbox, text, conf in zip(
            block.get("rec_polys", []),
            block.get("rec_texts", []),
            block.get("rec_scores", []),
        ):
            results.append((bbox, (str(text), float(conf))))

    print(f"  完成 ({elapsed:.2f}s) — 偵測到 {len(results)} 筆")
    for bbox, (text, conf) in results:
        print(f"    [{conf:.3f}] {text}")
    return results, elapsed


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EasyOCR vs PaddleOCR 視覺化比較工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video",  required=True,
                        help="影片路徑")
    parser.add_argument("--time",   required=True,
                        help="時間點: 秒 (30.5) 或 MM:SS (1:30) 或 HH:MM:SS (0:01:30)")
    parser.add_argument("--output", default=None,
                        help="輸出 PNG 路徑（預設: ocr_compare_<stem>_t<time>.png）")
    parser.add_argument("--lang",   nargs="+", default=["ch_sim", "en"],
                        help="EasyOCR 語言列表（預設: ch_sim en）")
    parser.add_argument("--no-gpu", action="store_true",
                        help="停用 GPU，改用 CPU")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"找不到影片: {video_path}")
        sys.exit(1)

    time_sec = parse_time(args.time)
    use_gpu  = not args.no_gpu

    print(f"影片: {video_path.name}")
    print(f"時間: {time_sec:.3f}s")

    # 1. extract frame
    print("\n[1/3] 擷取影格")
    frame = extract_frame(str(video_path), time_sec)
    if frame is None:
        sys.exit(1)
    print(f"  影格大小: {frame.shape[1]}x{frame.shape[0]}")

    # 2. EasyOCR
    print("\n[2/3] EasyOCR")
    easy_results, easy_elapsed = run_easyocr(frame, args.lang, use_gpu)

    # 3. PaddleOCR
    print("\n[3/3] PaddleOCR")
    paddle_results, paddle_elapsed = run_paddleocr(frame, use_gpu)

    # annotate frames
    easy_ann   = annotate_easyocr(frame, easy_results)
    paddle_ann = annotate_paddle(frame, paddle_results)

    easy_texts   = [(t, c) for _, t, c in easy_results]
    paddle_texts = [(t, c) for _, (t, c) in paddle_results]

    # build & save image
    safe_t   = args.time.replace(":", "-")
    stem     = f"ocr_compare_{video_path.stem}_t{safe_t}"
    out_img  = Path(args.output) if args.output else Path(f"{stem}.png")
    out_json = out_img.with_suffix(".json")

    print("\n合成比較圖...")
    comparison = build_comparison(
        frame, easy_ann, paddle_ann,
        easy_texts, paddle_texts,
        time_sec, video_path.name,
        easy_elapsed, paddle_elapsed,
    )
    cv2.imwrite(str(out_img), comparison)
    print(f"  圖片儲存: {out_img}")

    summary = {
        "video": str(video_path),
        "time_sec": time_sec,
        "easyocr": {
            "elapsed_sec": round(easy_elapsed, 3),
            "results": [
                {"text": t, "confidence": round(c, 4), "bbox": _bbox_to_jsonable(b)}
                for b, t, c in easy_results
            ],
        },
        "paddleocr": {
            "elapsed_sec": round(paddle_elapsed, 3),
            "results": [
                {"text": t, "confidence": round(c, 4), "bbox": _bbox_to_jsonable(b)}
                for b, (t, c) in paddle_results
            ],
        },
    }
    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  JSON 儲存: {out_json}")


if __name__ == "__main__":
    main()
