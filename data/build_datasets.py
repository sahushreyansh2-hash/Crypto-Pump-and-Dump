"""
build_datasets.py — Load, parse, and verify PD_logs_cleaned and pump_telegram datasets,
generating dataset statistics and preparing clean label files.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parse_datasets import parse_pd_logs, parse_pump_telegram, verify_and_analyze_datasets

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

LABELS_DIR = PROJECT_ROOT / "data" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_files():
    pd_logs_path = LABELS_DIR / "PD_logs_cleaned.json"
    pump_tele_path = LABELS_DIR / "pump_telegram.json"

    df_pd = parse_pd_logs(pd_logs_path) if pd_logs_path.exists() else pd.DataFrame()
    df_tele = parse_pump_telegram(pump_tele_path) if pump_tele_path.exists() else pd.DataFrame()

    return df_pd, df_tele

if __name__ == "__main__":
    df_pd, df_tele = load_dataset_files()
    logger.info(f"Loaded {len(df_pd)} PD logs records and {len(df_tele)} Telegram pump records.")
