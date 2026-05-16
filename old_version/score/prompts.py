SCORE_PROMPT_SINGLE = """\
這是一場羽球比賽的截圖，請找出畫面中記分板上的比分。

請只回傳以下 JSON 格式，不要任何其他文字或 markdown：
{
  "score_a": <上方隊伍的整數分數>,
  "score_b": <下方隊伍的整數分數>,
  "confidence": "<high | medium | low>",
  "note": "<備註，若無請填空字串>"
}

若完全看不到記分板，score_a 和 score_b 請填 null，confidence 填 "low"。\
"""

SCORE_PROMPT_BATCH = """\
以下有 {n} 張羽球比賽截圖，依序標記為圖片 1 到圖片 {n}，請分別辨識每張截圖中記分板的比分。

請只回傳以下 JSON 陣列（共 {n} 個元素，順序對應圖片編號），不要任何其他文字或 markdown：
[
  {{"image_index": 1, "score_a": <整數或null>, "score_b": <整數或null>, "confidence": "<high|medium|low>", "note": ""}},
  {{"image_index": 2, "score_a": <整數或null>, "score_b": <整數或null>, "confidence": "<high|medium|low>", "note": ""}},
  ...
]
若某張截圖完全看不到記分板，score_a 和 score_b 請填 null，confidence 填 "low"。\
"""

DETECT_BATCH_PROMPT = """\
以下有 {n} 張羽球比賽截圖，取自同一場比賽影片的不同時間段，依序標記為「圖 1」至「圖 {n}」。

任務：找出在「所有截圖中均清楚出現，且位置大致相同」的記分板（比分板）。

判斷標準：
1. 該記分板必須在每一張圖中都可見
2. 在所有圖中的位置大致固定（跨圖的邊框座標差異應在 0.1 以內）
3. 區域內明確顯示數字比分

請回傳此記分板在畫面中的相對座標（0.0~1.0，左上角為原點，x 向右、y 向下）。

只回傳以下 JSON，不要任何其他文字或 markdown：
{{
  "x1": <記分板左邊界 float>,
  "y1": <記分板上邊界 float>,
  "x2": <記分板右邊界 float>,
  "y2": <記分板下邊界 float>,
  "confidence": "<high|medium|low>",
  "consistent_count": <在幾張圖中看到此記分板，整數>,
  "note": ""
}}

若無法找到在所有圖中均穩定出現的記分板（位置漂移超過 0.1 或部分圖缺失），confidence 填 "low"，座標全填 null。\
"""

DETECT_PROMPT = """\
這是一場羽球比賽的截圖。請找出畫面中記分板（比分板）的確切位置，
以相對座標（0.0~1.0，左上角為原點，x 向右，y 向下）回傳邊界框。

請只回傳以下 JSON 格式，不要任何其他文字或 markdown：
{
  "x1": <記分板左邊界 0.0~1.0>,
  "y1": <記分板上邊界 0.0~1.0>,
  "x2": <記分板右邊界 0.0~1.0>,
  "y2": <記分板下邊界 0.0~1.0>,
  "confidence": "<high|medium|low>",
  "note": ""
}

若看不到記分板，所有座標值請填 null，confidence 填 "low"。\
"""
