#!/usr/bin/env python3
"""
Export a timeline-based viewer instead of concatenating clips.

This script reuses the rally detection from bre4.py and outputs:
1) segments CSV (start/end frame and time)
2) interactive HTML page with:
   - original video
   - clickable timeline marks
   - clickable segment list

Usage:
    python bre4_timeline.py [video.mp4] [analysis.csv] [output_dir]

Defaults:
    video.mp4   -> first .mp4 in current directory
    analysis.csv-> first .csv in current directory
    output_dir  -> <video_basename>_timeline
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

# 降低 FFmpeg 日誌級別，忽略非致命警告（如 h264 mmco 警告）
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|err_detect;ignore_err"

import cv2
import numpy as np
import pandas as pd

import bre4


def resolve_inputs() -> tuple[str, str, str]:
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    if not video_path:
        mp4s = sorted(glob.glob("*.mp4"))
        if not mp4s:
            raise FileNotFoundError("No .mp4 found in current directory")
        video_path = mp4s[0]

    if not csv_path:
        csvs = sorted(glob.glob("*.csv") + glob.glob("frame_diff*.csv"))
        if not csvs:
            raise FileNotFoundError("No .csv found in current directory")
        csv_path = csvs[0]

    if not output_dir:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"{base}_timeline"

    return video_path, csv_path, output_dir


def estimate_csv_fps(times: np.ndarray) -> float:
    valid = times[times > 0]
    if len(valid) < 2:
        return 30.0
    diffs = np.diff(valid)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 30.0
    return float(1.0 / np.median(diffs))


def get_video_duration(video_path: str, fallback_fps: float) -> tuple[float, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, fallback_fps

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    actual_fps = float(fps) if fps and fps > 0 else fallback_fps
    if frame_count and frame_count > 0 and actual_fps > 0:
        duration = float(frame_count / actual_fps)
    else:
        duration = 0.0

    return duration, actual_fps


def seconds_to_mmss(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def build_segment_rows(segments: list[tuple[int, int]], fps: float) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for i, (start_frame, end_frame) in enumerate(segments, start=1):
        start_sec = float(start_frame / fps)
        end_sec = float(end_frame / fps)
        rows.append(
            {
                "index": i,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "duration_sec": round(end_sec - start_sec, 4),
                "start_label": seconds_to_mmss(start_sec),
                "end_label": seconds_to_mmss(end_sec),
            }
        )
    return rows


def write_segments_csv(rows: list[dict[str, float | int]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "start_frame",
                "end_frame",
                "start_sec",
                "end_sec",
                "duration_sec",
                "start_label",
                "end_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def generate_html(video_src: str, rows: list[dict[str, float | int]], total_duration: float) -> str:
    rows_json = json.dumps(rows, ensure_ascii=False)
    safe_video_src = video_src.replace("\\", "/")

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rally Timeline Viewer</title>
  <style>
    :root {{
      --bg: #f4f6f0;
      --card: #ffffff;
      --ink: #1f2a1f;
      --muted: #5f6d5f;
      --accent: #d95d39;
      --accent-soft: #f8d8cd;
      --line: #d8e0d2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft JhengHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 0% 0%, #e8f1df 0, transparent 45%),
        radial-gradient(circle at 100% 100%, #f8e9d6 0, transparent 40%),
        var(--bg);
      padding: 24px;
    }}
    .wrap {{ max-width: 1040px; margin: 0 auto; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(32, 48, 26, 0.08);
    }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .meta {{ color: var(--muted); margin-bottom: 14px; }}
    video {{ width: 100%; border-radius: 12px; background: #000; }}
    .timeline {{
      margin-top: 14px;
      position: relative;
      height: 48px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: linear-gradient(90deg, #d9d0c8 0%, #d0c8bd 100%);
      overflow: hidden;
      cursor: pointer;
    }}
    .timeline::after {{
      content: '';
      position: absolute;
      left: 0;
      top: 0;
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #4a90e2 0%, #3d7cc4 100%);
      border-radius: 6px;
      pointer-events: none;
      z-index: 1;
      transition: width .05s linear;
    }}
    .segments-container {{
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      z-index: 2;
    }}
    .segment {{
      position: absolute;
      top: 0;
      height: 100%;
      border: none;
      cursor: pointer;
      transition: opacity .12s ease;
      opacity: 0.7;
      z-index: 3;
    }}
    .segment[data-color="0"] {{
      background: rgba(217, 93, 57, 0.7);
    }}
    .segment[data-color="1"] {{
      background: rgba(74, 144, 226, 0.7);
    }}
    .segment[data-color="2"] {{
      background: rgba(126, 211, 33, 0.7);
    }}
    .segment[data-color="3"] {{
      background: rgba(245, 166, 35, 0.7);
    }}
    .segment[data-color="4"] {{
      background: rgba(189, 16, 224, 0.7);
    }}
    .segment[data-color="5"] {{
      background: rgba(80, 227, 194, 0.7);
    }}
    .segment:hover {{
      opacity: 0.9;
    }}
    .segment.active {{
      opacity: 1;
      box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.6);
    }}
    .playhead {{
      position: absolute;
      left: 0;
      top: 0;
      width: 2px;
      height: 100%;
      background: #fff;
      box-shadow: 0 0 4px rgba(0, 0, 0, 0.3);
      pointer-events: none;
      z-index: 4;
    }}
    .zoom-controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      margin-bottom: 8px;
      font-size: 13px;
    }}
    .zoom-controls button {{
      width: 32px;
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
      font-weight: 600;
      color: var(--ink);
      transition: all .12s ease;
    }}
    .zoom-controls button:hover {{
      background: #f7faf3;
      border-color: #b9c9b4;
    }}
    .zoom-slider {{
      flex: 1;
      max-width: 200px;
      height: 5px;
      border-radius: 3px;
      background: #e8dcd2;
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    .zoom-slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #d95d39;
      cursor: pointer;
      box-shadow: 0 2px 6px rgba(217, 93, 57, 0.3);
    }}
    .zoom-slider::-moz-range-thumb {{
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #d95d39;
      border: none;
      cursor: pointer;
      box-shadow: 0 2px 6px rgba(217, 93, 57, 0.3);
    }}
    .zoom-label {{
      font-weight: 600;
      color: var(--muted);
      min-width: 45px;
    }}
    .timeline-container {{
      margin-top: 8px;
      overflow-x: auto;
      overflow-y: hidden;
      border-radius: 12px;
      border: 1px solid var(--line);
    }}
    .timeline-container::-webkit-scrollbar {{
      height: 8px;
    }}
    .timeline-container::-webkit-scrollbar-track {{
      background: #f4f6f0;
      border-radius: 4px;
    }}
    .timeline-container::-webkit-scrollbar-thumb {{
      background: #d8e0d2;
      border-radius: 4px;
    }}
    .timeline-container::-webkit-scrollbar-thumb:hover {{
      background: #c9d3bf;
    }}
    .legend {{ margin-top: 8px; color: var(--muted); font-size: 13px; }}
    .list {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 10px;
    }}
    .item {{
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      padding: 10px 12px;
      cursor: pointer;
    }}
    .item:hover {{ border-color: #b9c9b4; }}
    .item.active {{ border-color: var(--accent); box-shadow: 0 0 0 2px #f8d8cd inset; }}
    .item .t {{ font-weight: 600; }}
    .item .d {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Rally Timeline Viewer</h1>
      <div class="meta">點擊時間軸或下方片段列表可直接跳轉到原始影片該回合起點。</div>
      <video id="video" controls preload="metadata">
        <source src="{safe_video_src}" type="video/mp4" />
      </video>
      <div class="zoom-controls">
        <button id="zoomDecr" type="button" title="Zoom Out">−</button>
        <input type="range" id="zoomSlider" class="zoom-slider" min="50" max="300" value="100" />
        <span class="zoom-label" id="zoomLabel">100%</span>
        <button id="zoomIncr" type="button" title="Zoom In">+</button>
      </div>
      <div class="timeline-container">
        <div id="timeline" class="timeline">
          <div class="segments-container" id="segmentsContainer"></div>
          <div class="playhead" id="playhead"></div>
        </div>
      </div>
      <div class="legend">Highlighted blocks = detected rally segments</div>
      <div id="list" class="list"></div>
    </div>
  </div>

  <script>
    const segments = {rows_json};
    const totalDuration = {total_duration:.6f};

    const video = document.getElementById('video');
    const timeline = document.getElementById('timeline');
    const list = document.getElementById('list');
    const zoomSlider = document.getElementById('zoomSlider');
    const zoomLabel = document.getElementById('zoomLabel');
    const zoomDecr = document.getElementById('zoomDecr');
    const zoomIncr = document.getElementById('zoomIncr');
    const segmentsContainer = document.getElementById('segmentsContainer');
    const playhead = document.getElementById('playhead');
    
    let zoomLevel = 100; // percent

    function setZoom(level) {{
      zoomLevel = Math.max(50, Math.min(300, level));
      zoomSlider.value = zoomLevel;
      zoomLabel.textContent = zoomLevel + '%';
      timeline.style.width = zoomLevel + '%';
    }}

    function pct(t) {{
      if (totalDuration <= 0) return 0;
      return (t / totalDuration) * 100;
    }}

    function jumpTo(seg) {{
      video.currentTime = Number(seg.start_sec);
      video.play().catch(() => undefined);
      setActive(seg.index);
    }}

    function setActive(index) {{
      document.querySelectorAll('[data-seg]').forEach(el => {{
        const same = Number(el.dataset.seg) === Number(index);
        el.classList.toggle('active', same);
      }});
    }}

    function findActiveByTime(t) {{
      for (const seg of segments) {{
        if (t >= Number(seg.start_sec) && t <= Number(seg.end_sec)) return seg.index;
      }}
      return null;
    }}

    function updatePlayhead() {{
      if (totalDuration > 0) {{
        const pct_val = (video.currentTime / totalDuration) * 100;
        playhead.style.left = pct_val + '%';
      }}
    }}

    function render() {{
      if (!segments.length) {{
        timeline.innerHTML = '<div style="padding:12px;color:#5f6d5f;">No segments detected</div>';
        list.innerHTML = '<div style="color:#5f6d5f;">No segment list available</div>';
        return;
      }}

      for (const seg of segments) {{
        const block = document.createElement('button');
        block.type = 'button';
        block.className = 'segment';
        block.dataset.seg = seg.index;
        block.dataset.color = (seg.index - 1) % 6;
        block.style.left = `${{pct(seg.start_sec)}}%`;
        block.style.width = `${{Math.max(pct(seg.duration_sec), 0.3)}}%`;
        block.title = `#${{seg.index}} ${{seg.start_label}} - ${{seg.end_label}} (${{Number(seg.duration_sec).toFixed(2)}}s)`;
        block.addEventListener('click', e => {{ e.stopPropagation(); jumpTo(seg); }});
        segmentsContainer.appendChild(block);

        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'item';
        item.dataset.seg = seg.index;
        item.innerHTML = `
          <div class="t">#${{seg.index}} ${{seg.start_label}} - ${{seg.end_label}}</div>
          <div class="d">duration: ${{Number(seg.duration_sec).toFixed(2)}}s</div>
        `;
        item.addEventListener('click', () => jumpTo(seg));
        list.appendChild(item);
      }}
    }}

    video.addEventListener('timeupdate', () => {{
      updatePlayhead();
      const index = findActiveByTime(video.currentTime);
      if (index != null) setActive(index);
    }});

    video.addEventListener('seeking', updatePlayhead);

    timeline.addEventListener('click', e => {{
      const rect = timeline.getBoundingClientRect();
      const pct_val = (e.clientX - rect.left) / rect.width;
      if (pct_val >= 0 && pct_val <= 1) {{
        video.currentTime = totalDuration * pct_val;
        video.play().catch(() => undefined);
      }}
    }});

    zoomSlider.addEventListener('input', e => setZoom(Number(e.target.value)));
    zoomDecr.addEventListener('click', () => setZoom(zoomLevel - 20));
    zoomIncr.addEventListener('click', () => setZoom(zoomLevel + 20));

    render();
    setZoom(100);
    updatePlayhead();
  </script>
</body>
</html>
"""


def main() -> None:
    try:
        video_path, csv_path, output_dir = resolve_inputs()
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("bre4 Timeline Exporter")
    print("=" * 60)
    print(f"video     : {video_path}")
    print(f"csv       : {csv_path}")
    print(f"output dir: {output_dir}")

    df = pd.read_csv(csv_path)
    scores = df["Difference_Score"].values.astype(float)
    times = df["Time_Sec"].values.astype(float)
    fps = estimate_csv_fps(times)

    if bre4.MANUAL_THRESHOLD is not None:
        threshold = float(bre4.MANUAL_THRESHOLD)
        print(f"[detect] manual threshold={threshold:.0f}")
    else:
        threshold = bre4.find_threshold_gmm(scores)
        print(f"[detect] auto threshold={threshold:.0f}")

    segments = bre4.find_rally_segments(scores, fps, threshold)
    print(f"[detect] initial segments={len(segments)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] cannot open video: {video_path}")
        sys.exit(1)

    segments = bre4.verify_rally_segments(cap, segments, scores, fps)
    cap.release()
    print(f"[detect] verified segments={len(segments)}")

    duration_sec, actual_fps = get_video_duration(video_path, fps)
    if duration_sec <= 0 and len(times) > 0:
        duration_sec = float(times[-1])

    rows = build_segment_rows(segments, actual_fps if actual_fps > 0 else fps)

    csv_out = os.path.join(output_dir, "segments.csv")
    write_segments_csv(rows, csv_out)

    html_out = os.path.join(output_dir, "timeline_viewer.html")
    video_src = os.path.relpath(Path(video_path), Path(output_dir)).replace("\\", "/")
    html_text = generate_html(video_src=video_src, rows=rows, total_duration=duration_sec)
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_text)

    print("-" * 60)
    print(f"[done] segments : {len(rows)}")
    print(f"[done] csv      : {csv_out}")
    print(f"[done] html     : {html_out}")
    print("Open timeline_viewer.html in a browser to use clickable segment navigation.")


if __name__ == "__main__":
    main()
