import csv as csv_module
from pathlib import Path


def parse_csv_segments(csv_path: Path) -> list[dict]:
    required = {"Start_Frame", "End_Frame", "Start_Sec", "End_Sec"}
    segments = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少欄位: {missing}")
        for i, row in enumerate(reader):
            sf  = int(row["Start_Frame"])
            ef  = int(row["End_Frame"])
            ss  = float(row["Start_Sec"])
            es  = float(row["End_Sec"])
            dur = (
                float(row["Duration_Sec"])
                if "Duration_Sec" in row and row["Duration_Sec"]
                else es - ss
            )
            segments.append({
                "index":        i,
                "start_frame":  sf,
                "end_frame":    ef,
                "start_sec":    ss,
                "end_sec":      es,
                "duration_sec": dur,
            })
    return segments
