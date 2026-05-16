from pathlib import Path

CLIPS_DIR   = Path("output_clips")
OUTPUT_JSON = Path("score_labels.json")
API_KEY_ENV = "GEMINI_API_KEY"
MODEL       = "gemini-2.5-flash"

COMPRESS_W   = 1280
COMPRESS_H   = 720
JPEG_QUALITY = 85
MAX_CONCURRENT = 1
MAX_RETRIES    = 3

BATCH_SIZE              = 4
MIN_BINARY_INTERVAL_SEC = 5.0
RETRY_FRAME_POSITIONS   = (0.25, 0.75)

DETECT_SAMPLE_PCT   = 0.2   # fraction of segment count → total sample frames
DETECT_SAMPLE_MIN   = 9      # minimum samples (ensures ≥3 rounds × 3 frames)
DETECT_BATCH_SIZE   = 4      # images per detection round (3-5)
DETECT_BBOX_MARGIN  = 0.03

PRICE_INPUT_PER_M  = 0.075
PRICE_OUTPUT_PER_M = 0.30

TOKENS_IMAGE  = 1548
TOKENS_PROMPT = 150
TOKENS_OUTPUT = 80
