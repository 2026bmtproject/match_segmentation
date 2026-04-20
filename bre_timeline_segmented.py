#!/usr/bin/env python3
"""
Generate a timeline-based viewer from pre-segmented rally CSV.

This script reads a CSV file with pre-segmented rally data and outputs:
1) segments CSV (normalized format with labels)
2) interactive HTML page with:
   - original video
   - clickable timeline marks
   - clickable segment list

CSV Format (input):
    Start_Frame, End_Frame, Start_Sec, End_Sec, Duration_Sec

Usage:
    python bre_timeline_segmented.py [video.mp4] [segments.csv] [output_dir]
    python bre_timeline_segmented.py test2.mp4 test2_brepro_segments.csv

Defaults:
    video.mp4   -> first .mp4 in current directory
    segments.csv-> first .csv in current directory
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
import pandas as pd


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
        csvs = sorted(glob.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError("No .csv found in current directory")
        csv_path = csvs[0]

    if not output_dir:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"{base}_timeline"

    return video_path, csv_path, output_dir


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if not fps or fps <= 0 or not frame_count or frame_count <= 0:
        return 0.0

    return float(frame_count / fps)


def seconds_to_mmss(sec: float) -> str:
    """Convert seconds to MM:SS.FF format."""
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def build_segment_rows(df: pd.DataFrame) -> list[dict[str, float | int]]:
    """Convert dataframe rows to segment rows with formatted labels."""
    rows: list[dict[str, float | int]] = []
    
    for i, row in df.iterrows():
        start_frame = int(row["Start_Frame"])
        end_frame = int(row["End_Frame"])
        start_sec = float(row["Start_Sec"])
        end_sec = float(row["End_Sec"])
        duration_sec = float(row["Duration_Sec"])
        
        rows.append(
            {
                "index": i + 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "duration_sec": round(duration_sec, 4),
                "start_label": seconds_to_mmss(start_sec),
                "end_label": seconds_to_mmss(end_sec),
            }
        )
    return rows


def write_segments_csv(rows: list[dict[str, float | int]], output_path: str) -> None:
    """Write segments to CSV file."""
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
    """Generate interactive HTML timeline viewer."""
    rows_json = json.dumps(rows, ensure_ascii=False)
    safe_video_src = video_src.replace("\\", "/")

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rally Timeline Viewer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg: #0b0e0d;
      --surface: #131917;
      --card: #1a2020;
      --card2: #1f2826;
      --border: #263030;
      --border-hi: #354545;
      --text: #ddeae6;
      --muted: #5e7a72;
      --accent: #00e5b4;
      --accent-dim: rgba(0,229,180,0.15);
      --accent-glow: rgba(0,229,180,0.35);
      --orange: #ff6b3d;
      --orange-dim: rgba(255,107,61,0.15);
      --seg0: #00e5b4;
      --seg1: #ff6b3d;
      --seg2: #4db8ff;
      --seg3: #f0c040;
      --seg4: #c77dff;
      --seg5: #ff85a1;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'DM Sans', 'Microsoft JhengHei', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 28px 20px 48px;
      background-image:
        radial-gradient(ellipse 60% 40% at 10% 0%, rgba(0,229,180,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 40% 30% at 90% 100%, rgba(255,107,61,0.05) 0%, transparent 70%);
    }}
    .wrap {{ max-width: 1080px; margin: 0 auto; }}

    /* ── Header ── */
    .header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      margin-bottom: 24px;
      gap: 16px;
    }}
    .header-left {{ flex: 1; }}
    .eyebrow {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 6px;
    }}
    h1 {{
      font-family: 'Bebas Neue', sans-serif;
      font-size: clamp(32px, 5vw, 52px);
      letter-spacing: 0.04em;
      line-height: 1;
      color: var(--text);
    }}
    .subtitle {{
      margin-top: 6px;
      font-size: 13px;
      color: var(--muted);
    }}

    /* ── Stats pills ── */
    .stats {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .stat {{
      background: var(--card2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 14px;
      text-align: center;
    }}
    .stat-val {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 18px;
      font-weight: 600;
      color: var(--accent);
      line-height: 1;
    }}
    .stat-lbl {{
      font-size: 10px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-top: 4px;
    }}

    /* ── Main grid ── */
    .main-grid {{
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 16px;
      align-items: start;
    }}
    @media (max-width: 820px) {{
      .main-grid {{ grid-template-columns: 1fr; }}
    }}

    /* ── Panel ── */
    .panel {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
    }}
    .panel-header {{
      padding: 14px 18px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    .panel-title {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .panel-body {{ padding: 16px; }}

    /* ── Video ── */
    video {{
      width: 100%;
      display: block;
      border-radius: 0;
      background: #000;
      aspect-ratio: 16/9;
      object-fit: contain;
    }}
    .video-footer {{
      padding: 10px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border-top: 1px solid var(--border);
    }}
    .time-display {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      color: var(--accent);
      letter-spacing: 0.05em;
    }}
    .now-playing {{
      font-size: 12px;
      color: var(--muted);
    }}
    .now-playing span {{
      color: var(--text);
      font-weight: 600;
    }}

    /* ── Timeline ── */
    .tl-wrap {{
      padding: 0 16px 16px;
    }}
    .tl-toolbar {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 16px;
      border-bottom: 1px solid var(--border);
    }}
    .tl-label {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      flex: 1;
    }}
    .zoom-btn {{
      width: 28px;
      height: 28px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--card2);
      color: var(--text);
      font-size: 16px;
      line-height: 1;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all .15s ease;
    }}
    .zoom-btn:hover {{
      border-color: var(--accent);
      color: var(--accent);
      background: var(--accent-dim);
    }}
    .zoom-slider {{
      width: 100px;
      height: 4px;
      border-radius: 2px;
      background: var(--border-hi);
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    .zoom-slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 14px; height: 14px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      box-shadow: 0 0 8px var(--accent-glow);
    }}
    .zoom-slider::-moz-range-thumb {{
      width: 14px; height: 14px;
      border-radius: 50%;
      background: var(--accent);
      border: none;
      cursor: pointer;
      box-shadow: 0 0 8px var(--accent-glow);
    }}
    .zoom-pct {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: var(--muted);
      min-width: 40px;
      text-align: right;
    }}
    .timeline-scroll {{
      overflow-x: auto;
      overflow-y: hidden;
      margin-top: 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .timeline-scroll::-webkit-scrollbar {{ height: 6px; }}
    .timeline-scroll::-webkit-scrollbar-track {{ background: var(--surface); border-radius: 3px; }}
    .timeline-scroll::-webkit-scrollbar-thumb {{ background: var(--border-hi); border-radius: 3px; }}
    .timeline-scroll::-webkit-scrollbar-thumb:hover {{ background: var(--muted); }}
    .timeline {{
      position: relative;
      height: 56px;
      background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 9.09%,
        rgba(255,255,255,0.03) 9.09%,
        rgba(255,255,255,0.03) 9.1%
      ), linear-gradient(180deg, #1c2825 0%, #151e1b 100%);
      cursor: pointer;
      min-width: 100%;
    }}
    .segments-container {{
      position: absolute;
      inset: 0;
      z-index: 2;
    }}
    .segment {{
      position: absolute;
      top: 8px;
      height: calc(100% - 16px);
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: filter .12s ease, transform .12s ease;
      z-index: 3;
    }}
    .segment[data-color="0"] {{ background: var(--seg0); box-shadow: 0 0 10px rgba(0,229,180,0.4); }}
    .segment[data-color="1"] {{ background: var(--seg1); box-shadow: 0 0 10px rgba(255,107,61,0.4); }}
    .segment[data-color="2"] {{ background: var(--seg2); box-shadow: 0 0 10px rgba(77,184,255,0.4); }}
    .segment[data-color="3"] {{ background: var(--seg3); box-shadow: 0 0 10px rgba(240,192,64,0.4); }}
    .segment[data-color="4"] {{ background: var(--seg4); box-shadow: 0 0 10px rgba(199,125,255,0.4); }}
    .segment[data-color="5"] {{ background: var(--seg5); box-shadow: 0 0 10px rgba(255,133,161,0.4); }}
    .segment:hover {{ filter: brightness(1.3); transform: scaleY(1.1); }}
    .segment.active {{ filter: brightness(1.4); transform: scaleY(1.15); outline: 2px solid rgba(255,255,255,0.5); outline-offset: 1px; }}
    .playhead {{
      position: absolute;
      left: 0;
      top: 0;
      width: 2px;
      height: 100%;
      background: #fff;
      box-shadow: 0 0 6px rgba(255,255,255,0.6), 0 0 12px rgba(255,255,255,0.3);
      pointer-events: none;
      z-index: 5;
      transition: left .05s linear;
    }}
    .playhead::before {{
      content: '';
      position: absolute;
      top: -4px;
      left: 50%;
      transform: translateX(-50%);
      width: 0; height: 0;
      border-left: 5px solid transparent;
      border-right: 5px solid transparent;
      border-top: 6px solid #fff;
    }}
    .tl-hint {{
      padding: 6px 0 0;
      font-size: 11px;
      color: var(--muted);
      font-family: 'JetBrains Mono', monospace;
    }}

    /* ── Segment list ── */
    .list-panel {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      max-height: 620px;
      display: flex;
      flex-direction: column;
    }}
    .list-panel .panel-header {{
      flex-shrink: 0;
    }}
    .list-scroll {{
      overflow-y: auto;
      flex: 1;
      padding: 10px;
    }}
    .list-scroll::-webkit-scrollbar {{ width: 6px; }}
    .list-scroll::-webkit-scrollbar-track {{ background: transparent; }}
    .list-scroll::-webkit-scrollbar-thumb {{ background: var(--border-hi); border-radius: 3px; }}
    .item {{
      width: 100%;
      text-align: left;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--card2);
      padding: 10px 12px;
      cursor: pointer;
      transition: all .15s ease;
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 6px;
    }}
    .item:last-child {{ margin-bottom: 0; }}
    .item:hover {{
      border-color: var(--border-hi);
      background: #243030;
      transform: translateX(2px);
    }}
    .item.active {{
      border-color: var(--accent);
      background: var(--accent-dim);
      transform: translateX(3px);
    }}
    .item-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
    }}
    .item-body {{ flex: 1; min-width: 0; }}
    .item-num {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
      color: var(--muted);
      margin-bottom: 2px;
    }}
    .item-time {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .item-dur {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: var(--muted);
      flex-shrink: 0;
    }}
    .item.active .item-dur {{ color: var(--accent); }}

    /* ── Dot color variants ── */
    .dot-0 {{ background: var(--seg0); box-shadow: 0 0 5px var(--seg0); }}
    .dot-1 {{ background: var(--seg1); box-shadow: 0 0 5px var(--seg1); }}
    .dot-2 {{ background: var(--seg2); box-shadow: 0 0 5px var(--seg2); }}
    .dot-3 {{ background: var(--seg3); box-shadow: 0 0 5px var(--seg3); }}
    .dot-4 {{ background: var(--seg4); box-shadow: 0 0 5px var(--seg4); }}
    .dot-5 {{ background: var(--seg5); box-shadow: 0 0 5px var(--seg5); }}

    /* ── Animated entry ── */
    @keyframes fadeUp {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .item {{ animation: fadeUp .25s ease both; }}
  </style>
</head>
<body>
  <div class="wrap">

    <!-- Header -->
    <div class="header">
      <div class="header-left">
        <div class="eyebrow">Video Analysis</div>
        <h1>Rally Timeline Viewer</h1>
        <div class="subtitle">點擊時間軸或右側列表，即可跳轉到對應回合起點</div>
      </div>
      <div class="stats" id="statsBar"></div>
    </div>

    <!-- Main grid: video+timeline | list -->
    <div class="main-grid">

      <!-- Left column -->
      <div>
        <!-- Video panel -->
        <div class="panel">
          <div class="panel-header">
            <span class="panel-title">Source Video</span>
            <span class="time-display" id="timeDisplay">00:00.00</span>
          </div>
          <video id="video" controls preload="metadata">
            <source src="{safe_video_src}" type="video/mp4" />
          </video>
          <div class="video-footer">
            <div class="now-playing" id="nowPlaying">No rally active</div>
          </div>
        </div>

        <!-- Timeline panel -->
        <div class="panel" style="margin-top:12px;">
          <div class="tl-toolbar">
            <span class="tl-label">Timeline</span>
            <button class="zoom-btn" id="zoomDecr" title="Zoom Out">−</button>
            <input type="range" class="zoom-slider" id="zoomSlider" min="50" max="500" value="100" />
            <span class="zoom-pct" id="zoomLabel">100%</span>
            <button class="zoom-btn" id="zoomIncr" title="Zoom In">+</button>
          </div>
          <div class="tl-wrap">
            <div class="timeline-scroll">
              <div id="timeline" class="timeline">
                <div class="segments-container" id="segmentsContainer"></div>
                <div class="playhead" id="playhead"></div>
              </div>
            </div>
            <div class="tl-hint">▲ Highlighted blocks = rally segments · Click to seek</div>
          </div>
        </div>
      </div>

      <!-- Right column: segment list -->
      <div class="list-panel">
        <div class="panel-header">
          <span class="panel-title">Segments</span>
          <span class="panel-title" id="activeLabel">—</span>
        </div>
        <div class="list-scroll">
          <div id="list"></div>
        </div>
      </div>

    </div>
  </div>

  <script>
    const segments = {rows_json};
    const totalDuration = {total_duration:.6f};

    const video     = document.getElementById('video');
    const timeline  = document.getElementById('timeline');
    const list      = document.getElementById('list');
    const playhead  = document.getElementById('playhead');
    const segsCont  = document.getElementById('segmentsContainer');
    const zoomSlider = document.getElementById('zoomSlider');
    const zoomLabel  = document.getElementById('zoomLabel');
    const timeDisplay = document.getElementById('timeDisplay');
    const nowPlaying  = document.getElementById('nowPlaying');
    const activeLabel = document.getElementById('activeLabel');
    const statsBar    = document.getElementById('statsBar');

    let zoomLevel = 100;
    const SEG_COLORS = ['seg0','seg1','seg2','seg3','seg4','seg5'];

    /* ── Stats ── */
    function buildStats() {{
      if (!segments.length) return;
      const totalRally = segments.reduce((s, r) => s + Number(r.duration_sec), 0);
      const avg = totalRally / segments.length;
      function mmss(s) {{
        const m = Math.floor(s/60);
        return String(m).padStart(2,'0') + ':' + (s - m*60).toFixed(2).padStart(5,'0');
      }}
      const data = [
        {{ val: segments.length, lbl: 'Rallies' }},
        {{ val: mmss(totalRally), lbl: 'Total Rally' }},
        {{ val: avg.toFixed(1) + 's', lbl: 'Avg Length' }},
        {{ val: mmss(totalDuration), lbl: 'Video Length' }},
      ];
      statsBar.innerHTML = data.map(d =>
        `<div class="stat"><div class="stat-val">${{d.val}}</div><div class="stat-lbl">${{d.lbl}}</div></div>`
      ).join('');
    }}

    /* ── Zoom ── */
    function setZoom(level) {{
      zoomLevel = Math.max(50, Math.min(500, level));
      zoomSlider.value = zoomLevel;
      zoomLabel.textContent = zoomLevel + '%';
      timeline.style.width = zoomLevel + '%';
    }}

    /* ── Helpers ── */
    function pct(t) {{ return totalDuration > 0 ? (t / totalDuration) * 100 : 0; }}

    function fmtTime(s) {{
      const m = Math.floor(s / 60);
      return String(m).padStart(2,'0') + ':' + (s - m*60).toFixed(2).padStart(5,'0');
    }}

    function jumpTo(seg) {{
      video.currentTime = Number(seg.start_sec);
      video.play().catch(() => {{}});
      setActive(seg.index);
    }}

    function setActive(index) {{
      document.querySelectorAll('[data-seg]').forEach(el => {{
        el.classList.toggle('active', Number(el.dataset.seg) === Number(index));
      }});
      if (index != null) {{
        const seg = segments.find(s => s.index === Number(index));
        if (seg) {{
          activeLabel.textContent = '#' + seg.index;
          nowPlaying.innerHTML = `Rally <span>#${{seg.index}}</span> · ${{fmtTime(Number(seg.start_sec))}} → ${{fmtTime(Number(seg.end_sec))}}`;
          // scroll list item into view
          const el = document.querySelector(`.item[data-seg="${{index}}"]`);
          if (el) el.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
        }}
      }} else {{
        activeLabel.textContent = '—';
        nowPlaying.textContent = 'No rally active';
      }}
    }}

    function findActiveByTime(t) {{
      for (const seg of segments) {{
        if (t >= Number(seg.start_sec) && t <= Number(seg.end_sec)) return seg.index;
      }}
      return null;
    }}

    function updatePlayhead() {{
      if (totalDuration > 0) {{
        playhead.style.left = (video.currentTime / totalDuration * 100) + '%';
      }}
      timeDisplay.textContent = fmtTime(video.currentTime);
    }}

    /* ── Render ── */
    function render() {{
      if (!segments.length) {{
        timeline.innerHTML = '<div style="padding:16px;color:var(--muted);font-size:13px;">No segments detected</div>';
        list.innerHTML = '<div style="padding:16px;color:var(--muted);font-size:13px;">No segment list available</div>';
        return;
      }}

      segments.forEach((seg, i) => {{
        const colorIdx = (seg.index - 1) % 6;

        // Timeline block
        const block = document.createElement('button');
        block.type = 'button';
        block.className = 'segment';
        block.dataset.seg = seg.index;
        block.dataset.color = colorIdx;
        block.style.left  = `${{pct(seg.start_sec)}}%`;
        block.style.width = `${{Math.max(pct(seg.duration_sec), 0.4)}}%`;
        block.title = `#${{seg.index}}  ${{seg.start_label}} – ${{seg.end_label}}  (${{Number(seg.duration_sec).toFixed(2)}}s)`;
        block.addEventListener('click', e => {{ e.stopPropagation(); jumpTo(seg); }});
        segsCont.appendChild(block);

        // List item
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'item';
        item.dataset.seg = seg.index;
        item.style.animationDelay = (i * 0.03) + 's';
        item.innerHTML = `
          <div class="item-dot dot-${{colorIdx}}"></div>
          <div class="item-body">
            <div class="item-num">Rally #${{seg.index}}</div>
            <div class="item-time">${{seg.start_label}} – ${{seg.end_label}}</div>
          </div>
          <div class="item-dur">${{Number(seg.duration_sec).toFixed(1)}}s</div>
        `;
        item.addEventListener('click', () => jumpTo(seg));
        list.appendChild(item);
      }});
    }}

    /* ── Events ── */
    video.addEventListener('timeupdate', () => {{
      updatePlayhead();
      const idx = findActiveByTime(video.currentTime);
      setActive(idx);
    }});
    video.addEventListener('seeking', updatePlayhead);

    timeline.addEventListener('click', e => {{
      const rect = timeline.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width;
      if (ratio >= 0 && ratio <= 1) {{
        video.currentTime = totalDuration * ratio;
        video.play().catch(() => {{}});
      }}
    }});

    zoomSlider.addEventListener('input', e => setZoom(Number(e.target.value)));
    document.getElementById('zoomDecr').addEventListener('click', () => setZoom(zoomLevel - 25));
    document.getElementById('zoomIncr').addEventListener('click', () => setZoom(zoomLevel + 25));

    buildStats();
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
    print("Timeline Viewer Generator (Pre-segmented)")
    print("=" * 60)
    print(f"video     : {video_path}")
    print(f"csv       : {csv_path}")
    print(f"output dir: {output_dir}")

    # Read pre-segmented CSV
    try:
        df = pd.read_csv(csv_path)
        required_cols = {"Start_Frame", "End_Frame", "Start_Sec", "End_Sec", "Duration_Sec"}
        if not required_cols.issubset(df.columns):
            print(f"[error] CSV missing required columns: {required_cols}")
            sys.exit(1)
    except Exception as exc:
        print(f"[error] Failed to read CSV: {exc}")
        sys.exit(1)

    # Verify video file
    if not os.path.exists(video_path):
        print(f"[error] Video file not found: {video_path}")
        sys.exit(1)

    # Get video duration
    duration_sec = get_video_duration(video_path)
    if duration_sec <= 0:
        print(f"[warning] Could not determine video duration")
        duration_sec = float(df["End_Sec"].max())

    # Build segment rows
    rows = build_segment_rows(df)
    print(f"[load] segments={len(rows)}")

    # Write segments CSV
    csv_out = os.path.join(output_dir, "segments.csv")
    write_segments_csv(rows, csv_out)

    # Generate and write HTML
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