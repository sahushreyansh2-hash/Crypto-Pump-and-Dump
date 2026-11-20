"""
convert_ocr_to_json.py — Script to extract JSON lines from OCR transcript and save raw dataset files.
"""

from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

pd_logs_file = LABELS_DIR / "PD_logs_cleaned.json"
pump_tele_file = LABELS_DIR / "pump_telegram.json"

print(f"Target dataset directory: {LABELS_DIR.resolve()}")
