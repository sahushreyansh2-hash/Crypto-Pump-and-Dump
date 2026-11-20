"""
process_raw_logs.py — Extract OCR dataset contents from prompt into clean dataset files
and run schema verification.
"""

import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Function to parse OCR JSON lines from raw text
def extract_json_objects(text: str) -> list[dict]:
    records = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Find JSON boundaries
        start = line.find("{")
        end = line.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = line[start : end + 1]
            try:
                obj = json.loads(json_str)
                records.append(obj)
            except json.JSONDecodeError:
                pass
    return records
