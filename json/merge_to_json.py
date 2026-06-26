#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_to_json.py
把 MK_vs_CT_2019 的多來源偵測資料合併成給 LLM 寫賽評用的 JSON。

輸出:
    out/match_meta.json
    out/set1.json / set2.json / set3.json

資料來源與規則:
    startframe.csv   每段 [Start_Frame, End_Frame]（global frame），決定 shot 屬於哪一段
    event/*.csv      球軌跡 X,Y + Hit==1（偵測到的擊球時刻）
    skeleton/*.csv   每 frame top/bottom 兩名球員關節 → 用雙腳踝中點當站位
    score.csv        每段比分（已確認皆準，忽略 confidence）
    court.csv        16 個校正點 → 前 4 點為球場四角，建單應性轉 0~1 球場座標
    setN.csv         **只使用 frame_num 對應 type**，其餘欄位一律不取

座標/場地:
    set1 底端=A 頂端=B / set2 底端=B 頂端=A(輪換) / set3 底端=A 頂端=B
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
import cv2

# ----------------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
NAME = "MK_vs_CT_2019"
OUT  = BASE / "out"
FPS  = 25.0

PLAYERS = {"A": "MK", "B": "CT"}

# 全場第一球的發球者 (開賽猜邊決定，不在任何偵測資料中，需外部給定)。
# 其餘所有發球者皆由羽球規則「上一回合勝方發下一球」+ score.csv 勝負自動推得；
# 每盤第一球發球者 = 前一盤勝方。
GLOBAL_FIRST_SERVER = "A"

# 每個 set 的場地方位 (skeleton 的 top/bottom -> 球員 A/B)
COURT_SWITCH = {
    1: {"bottom": "A", "top": "B"},
    2: {"bottom": "B", "top": "A"},
    3: {"bottom": "A", "top": "B"},
}

# 降採樣 (保留軌跡)
BALL_EVERY   = 4   # 球軌跡每幾 frame 取一點 (~6 fps)；命中 frame 一律保留
PLAYER_EVERY = 6   # 球員軌跡每幾 frame 取一點 (~4 fps)

HIT_MATCH_TOL = 20  # set 的一拍 frame_num 對應到偵測 Hit 的容忍 (frame)
SNAP_TOL      = 30  # shot 的 frame_num 落在段邊界外但 <= 此值 → 吸附到最近段
SKEL_WIN      = 6   # 擊球 frame 無骨架時，往前後 <= 此值 frame 找最近骨架


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
def seg_name(idx: int) -> str:
    return f"{NAME}_seg{idx:04d}"


def r(v, n=1):
    """四捨五入並轉成 int(若整數) / float。"""
    if v is None:
        return None
    f = round(float(v), n)
    return int(f) if f == int(f) else f


# ----------------------------------------------------------------------------
# 載入各來源
# ----------------------------------------------------------------------------
def load_startframes():
    """回傳 {seg_idx: (start_frame, end_frame)}；檔案列順序即 seg0001..segNNNN。"""
    path = BASE / f"{NAME}_startframe.csv"
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows, start=1):
        out[i] = (int(float(row["Start_Frame"])), int(float(row["End_Frame"])))
    return out


def load_scores():
    """回傳 {seg_idx: (score_a, score_b)}（忽略 confidence）。"""
    path = BASE / f"{NAME}_score.csv"
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            m = re.search(r"seg(\d+)", row["file"])
            if not m:
                continue
            out[int(m.group(1))] = (int(row["score_a"]), int(row["score_b"]))
    return out


def load_court_homography():
    """前 4 點 = 球場四角 (TL, TR, BL, BR)，建像素 -> 單位球場座標的單應性。
    球場座標: nx 0=左 1=右；ny 0=遠端(top) 1=近端(bottom)。"""
    path = BASE / f"{NAME}_court.csv"
    pts = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = line.split(";")
            pts.append([float(x), float(y)])
    src = np.array(pts[:4], dtype=np.float32)              # TL, TR, BL, BR
    dst = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H, pts


def load_set_shots(set_no, startframes):
    """讀 setN.csv，**只取 frame_num 與 type**，依 frame_num 落在哪段分組。
    回傳 {seg_idx: [(frame_num, type), ...]}（已按 frame_num 排序）。"""
    path = BASE / f"{NAME}_set{set_no}.csv"
    grouped = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fn = row.get("frame_num")
            typ = row.get("type")
            if not fn or not typ:
                continue
            fn = int(float(fn))
            seg = None
            for idx, (s, e) in startframes.items():
                if s <= fn <= e:
                    seg = idx
                    break
            if seg is None:                       # 落在段邊界縫隙 → 吸附最近段
                best, bd = None, None
                for idx, (s, e) in startframes.items():
                    d = min(abs(s - fn), abs(e - fn))
                    if bd is None or d < bd:
                        best, bd = idx, d
                if bd is not None and bd <= SNAP_TOL:
                    seg = best
            if seg is None:
                continue
            grouped.setdefault(seg, []).append((fn, typ.strip()))
    for seg in grouped:
        grouped[seg].sort(key=lambda t: t[0])
    return grouped


def load_event(seg_idx):
    """單段 event：回傳 dict local_frame -> (visibility, x, y, hit)。"""
    path = BASE / f"{NAME}_event" / f"{seg_name(seg_idx)}_event.csv"
    frames = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fr = int(float(row["Frame"]))
            frames[fr] = (int(float(row["Visibility"])), float(row["X"]),
                          float(row["Y"]), int(float(row["Hit"])))
    return frames


def _kp(row, name):
    """取關節 (x,y)；缺失回 None。"""
    try:
        x, y = float(row[f"{name}_x"]), float(row[f"{name}_y"])
    except (KeyError, ValueError, TypeError):
        return None
    if x == 0 and y == 0:
        return None
    return (x, y)


def _player_center(row):
    """雙腳踝中點；缺則退而求其次用髖中點 / bbox 底中。"""
    la, ra = _kp(row, "L_ankle"), _kp(row, "R_ankle")
    if la and ra:
        return ((la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2)
    if la or ra:
        p = la or ra
        return p
    lh, rh = _kp(row, "L_hip"), _kp(row, "R_hip")
    if lh and rh:
        return ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    try:
        return ((float(row["bbox_x1"]) + float(row["bbox_x2"])) / 2, float(row["bbox_y2"]))
    except (KeyError, ValueError):
        return None


def load_skeleton(seg_idx):
    """單段 skeleton：回傳 dict local_frame -> {'top': center, 'bottom': center}。"""
    path = BASE / f"{NAME}_skeleton" / f"{seg_name(seg_idx)}_skeleton.csv"
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fr = int(float(row["frame"]))
            who = row["player"].strip()
            if who not in ("top", "bottom"):
                continue
            slot = out.setdefault(fr, {})
            if who not in slot:                # 每 frame 每側取第一筆
                c = _player_center(row)
                if c:
                    slot[who] = c
    return out


def load_event_global(seg_list, startframes):
    """跨段合併 event 到 global frame 座標：dict global_frame -> (vis,x,y,hit), hits[]。"""
    frames, hits = {}, []
    for seg in seg_list:
        s = startframes[seg][0]
        for fr, val in load_event(seg).items():
            g = s + fr
            frames[g] = val
            if val[3] == 1:
                hits.append(g)
    hits.sort()
    return frames, hits


def load_skeleton_global(seg_list, startframes):
    """跨段合併 skeleton 到 global frame 座標。"""
    out = {}
    for seg in seg_list:
        s = startframes[seg][0]
        for fr, slot in load_skeleton(seg).items():
            out[s + fr] = slot
    return out


def group_rallies(seg_indices, scores):
    """把同一回合被切成的多段合併。判據用 score.csv:
    一盤內每個回合開打前的比分 (a,b) 互不重複 (每球必有一方得分，分數單調遞增)，
    所以『連續且比分相同的段』必為同一回合被切開 → 合併。
    回傳 [[seg_idx,...], ...]，每個子串列是一個 rally。"""
    seg_indices = sorted(seg_indices)
    groups, cur = [], []
    for idx in seg_indices:
        if cur and scores.get(idx) == scores.get(cur[-1]):
            cur.append(idx)
        else:
            if cur:
                groups.append(cur)
            cur = [idx]
    if cur:
        groups.append(cur)
    return groups


# ----------------------------------------------------------------------------
# 球場座標 / 語意化
# ----------------------------------------------------------------------------
def make_court_fns(H):
    def to_court(x, y):
        if x is None or y is None:
            return None
        p = np.array([[[float(x), float(y)]]], dtype=np.float32)
        nx, ny = cv2.perspectiveTransform(p, H)[0, 0]
        return (float(nx), float(ny))

    def zone(nx, ny):
        # 左右
        lat = "left" if nx < 0.34 else ("right" if nx > 0.66 else "center")
        # 遠端(top)/近端(bottom)
        half = "far" if ny < 0.5 else "near"
        # 距網深度 (0=貼網, 1=底線)
        d = abs(ny - 0.5) * 2
        depth = "frontcourt" if d < 0.34 else ("backcourt" if d > 0.66 else "midcourt")
        return f"{half}-{depth}-{lat}"

    def describe(x, y):
        c = to_court(x, y)
        if c is None:
            return None
        nx, ny = c
        return {
            "px": [r(x), r(y)],
            "court": [r(nx, 3), r(ny, 3)],
            "zone": zone(nx, ny),
        }

    return to_court, zone, describe


# ----------------------------------------------------------------------------
# 單段 (rally) 處理
# ----------------------------------------------------------------------------
def match_shots_to_hits(set_shots, hits):
    """把 set 每一拍 (global frame_num) 對到偵測 Hit (global frame)。
    回傳 [(anchor_global_frame, type, matched_hit_bool), ...]。"""
    used = set()
    result = []
    for fn, typ in set_shots:
        best, best_d = None, None
        for h in hits:
            if h in used:
                continue
            d = abs(h - fn)
            if best_d is None or d < best_d:
                best, best_d = h, d
        if best is not None and best_d <= HIT_MATCH_TOL:
            used.add(best)
            result.append((best, typ, True))
        else:
            result.append((fn, typ, False))      # 無對應偵測 Hit，仍以 frame_num 定位
    result.sort(key=lambda t: t[0])
    return result


OTHER = {"top": "bottom", "bottom": "top"}


def skel_at(skel, frame, win=SKEL_WIN):
    """擊球 frame 無骨架時，往前後找最近一筆 (補偵測缺漏)。"""
    if frame in skel:
        return skel[frame]
    for d in range(1, win + 1):
        if frame - d in skel:
            return skel[frame - d]
        if frame + d in skel:
            return skel[frame + d]
    return {}


def build_rally(seg_list, set_no, set_shots, server, startframes, describe):
    """seg_list 為合併後屬於同一回合的段；一律以 global frame 計算。
    server (A/B) = 本回合發球者，由規則推得；單打首拍即發球者，之後嚴格輪流。"""
    rally_start = startframes[seg_list[0]][0]
    rally_end = startframes[seg_list[-1]][1]
    ev_frames, hits = load_event_global(seg_list, startframes)
    skel = load_skeleton_global(seg_list, startframes)
    orient = COURT_SWITCH[set_no]            # {'bottom':?, 'top':?}

    def rel_t(g):
        return round((g - rally_start) / FPS, 2)

    # ---- 逐拍 (單打嚴格輪流擊球；發球者所在側由規則決定) ----
    matched = match_shots_to_hits(set_shots, hits)
    start_side = next(side for side, pid in orient.items() if pid == server)
    shots = []
    for i, (frame, typ, is_hit) in enumerate(matched, start=1):
        ev = ev_frames.get(frame)
        ball = (ev[1], ev[2]) if ev else (None, None)
        sk = skel_at(skel, frame)

        hitter_side = start_side if i % 2 == 1 else OTHER[start_side]
        opp_side = OTHER[hitter_side]
        player = orient[hitter_side]
        hitter_c = sk.get(hitter_side)
        opp_c = sk.get(opp_side)

        shots.append({
            "shot_no": i,
            "t": rel_t(frame),
            "player": player,                         # 擊球者 (A/B)
            "type": typ,                              # ← 唯一來自 set.csv
            "detected_hit": is_hit,
            # 球在空中，投影到地面無意義 → 擊球點只給像素座標
            "hit_px": [r(ball[0]), r(ball[1])] if ball[0] is not None else None,
            # 球員雙腳在地面 → 給球場座標與區域語意
            "player_pos": describe(*hitter_c) if hitter_c else None,
            "opponent_pos": describe(*opp_c) if opp_c else None,
        })

    # ---- 球軌跡 (降採樣，命中必留) ----
    ball_track = []
    hitset = set(hits)
    for g in sorted(ev_frames):
        vis, x, y, hit = ev_frames[g]
        if vis != 1:
            continue
        if (g - rally_start) % BALL_EVERY == 0 or g in hitset:
            pt = {"t": rel_t(g), "x": r(x), "y": r(y)}
            if g in hitset:
                pt["tag"] = "hit"
            ball_track.append(pt)

    # ---- 球員軌跡 (降採樣) ----
    player_track = {"A": [], "B": []}
    for g in sorted(skel):
        if (g - rally_start) % PLAYER_EVERY != 0:
            continue
        for side, c in skel[g].items():
            pid = orient.get(side)
            if pid:
                player_track[pid].append({"t": rel_t(g), "x": r(c[0]), "y": r(c[1])})

    return {
        "segments": [seg_name(s) for s in seg_list],
        "frame_range": [rally_start, rally_end],
        "duration_s": round((rally_end - rally_start) / FPS, 1),
        "shot_count": len(shots),
        "shots": shots,
        "ball_track": ball_track,
        "player_track": player_track,
    }


# ----------------------------------------------------------------------------
# 比分推導 (僅用 score.csv)
# ----------------------------------------------------------------------------
def resolve_scores(groups, scores):
    """groups 為合併後的回合 (每個是段索引串列)。比分取各回合「首段」的 score.csv。
    回傳 list of (score_before, score_after, winner) 對齊 groups，及 set 終分。"""
    out = []
    n = len(groups)
    set_final = None
    for i, group in enumerate(groups):
        before = scores.get(group[0], (None, None))
        if i + 1 < n:
            after = scores.get(groups[i + 1][0], before)
        else:
            # set 最後一回合：score.csv 不含賽末點後的比分 → 由領先方 +1 推得
            a, b = before
            after = (a + 1, b) if a >= b else (a, b + 1)
            set_final = after
        winner = None
        if before[0] is not None:
            if after[0] > before[0]:
                winner = "A"
            elif after[1] > before[1]:
                winner = "B"
        out.append((list(before), list(after), winner))
    return out, set_final


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    startframes = load_startframes()
    scores = load_scores()
    H, court_pts = load_court_homography()
    _, _, describe = make_court_fns(H)

    set_finals = {}
    prev_set_winner = None                       # 前一盤勝方 → 本盤首發者
    for set_no in (1, 2, 3):
        grouped = load_set_shots(set_no, startframes)
        groups = group_rallies(grouped.keys(), scores)
        score_info, set_final = resolve_scores(groups, scores)
        set_finals[set_no] = set_final

        # 發球序：首球 = 前一盤勝方(第一盤為 GLOBAL_FIRST_SERVER)；其餘 = 上一回合勝方
        first_server = prev_set_winner or GLOBAL_FIRST_SERVER
        servers = [first_server]
        for before, after, winner in score_info[:-1]:
            servers.append(winner or servers[-1])    # 勝方發下一球(罕見平手沿用)

        rallies = []
        for rno, group in enumerate(groups, start=1):
            set_shots = sorted((s for seg in group for s in grouped[seg]),
                               key=lambda t: t[0])
            rally = build_rally(group, set_no, set_shots, servers[rno - 1],
                                startframes, describe)
            before, after, winner = score_info[rno - 1]
            rally = {
                "rally_no": rno,
                **rally,
                "score_before": before,
                "score_after": after,
                "winner": winner,
            }
            rallies.append(rally)

        set_json = {
            "match": NAME,
            "set_no": set_no,
            "fps": FPS,
            "court_orientation": COURT_SWITCH[set_no],
            "final_score": set_final,
            "rally_count": len(rallies),
            "rallies": rallies,
        }
        path = OUT / f"set{set_no}.json"
        path.write_text(json.dumps(set_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  set{set_no}: {len(rallies)} rallies  final={set_final}  -> {path.name}")

        sf = set_final or (0, 0)
        prev_set_winner = "A" if sf[0] > sf[1] else "B"

    # ---- match_meta ----
    sets_final = [set_finals[s] for s in (1, 2, 3)]
    a_sets = sum(1 for f in sets_final if f and f[0] > f[1])
    b_sets = sum(1 for f in sets_final if f and f[1] > f[0])
    meta = {
        "name": NAME,
        "fps": FPS,
        "players": PLAYERS,
        "court_switch": COURT_SWITCH,
        "final_score": {
            "sets": sets_final,
            "sets_won": {"A": a_sets, "B": b_sets},
            "winner": "A" if a_sets > b_sets else "B",
        },
        "court_calibration": {
            "image_points": court_pts,
            "outer_corners_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
            "note": "球場座標 nx:0=左1=右, ny:0=遠端(top)1=近端(bottom)；網約在 ny=0.5",
        },
        "data_sources": {
            "ball_track / hit_pos": "event/*.csv",
            "player positions": "skeleton/*.csv (ankle midpoint) + court_switch",
            "scores / winner": "score.csv (+ 終分由領先方推得)",
            "court coords": "court.csv homography",
            "shot type": "setN.csv (frame_num→type only)",
        },
    }
    (OUT / "match_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  match winner: {meta['final_score']['winner']}  sets {a_sets}-{b_sets}  -> match_meta.json")


if __name__ == "__main__":
    print("Merging ...")
    main()
    print("Done. 輸出在 out/")
