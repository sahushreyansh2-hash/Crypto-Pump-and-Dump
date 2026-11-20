"""
extract_datasets.py — Save raw datasets into data/labels/ and execute schema verification.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parse_datasets import parse_pd_logs, parse_pump_telegram, verify_and_analyze_datasets

DATA_DIR = PROJECT_ROOT / "data" / "labels"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Parse logic test & dataset verification runner
print("Data Extraction & Schema Verification Script Initialized.")
