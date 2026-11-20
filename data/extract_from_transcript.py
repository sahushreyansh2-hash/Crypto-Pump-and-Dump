"""
extract_from_transcript.py — Extracts the dataset JSON records directly from the transcript log.
"""

import json
import re
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRANSCRIPT_PATH = Path(r"C:\Users\shrey\.gemini\antigravity\brain\24528497-c421-487d-9a34-cccafab930b9\.system_generated\logs\transcript_full.jsonl")
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

def extract():
    print(f"Reading transcript from {TRANSCRIPT_PATH}...")
    if not TRANSCRIPT_PATH.exists():
        print(f"Transcript file not found at {TRANSCRIPT_PATH}")
        return

    pd_records = []
    telegram_records = []

    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if "channel_id" in line:
                # Matches records from PD logs: {"channel_id": ...}
                matches = re.findall(r'\{"channel_id":[^}]+\}', line)
                for m in matches:
                    try:
                        obj = json.loads(m)
                        if "coin" in obj and "timestamp" in obj:
                            pd_records.append(obj)
                    except Exception:
                        pass

            if "group" in line and "date" in line:
                # Matches records from Pump Telegram: {"symbol": ..., "group": ..., "date": ..., "hour": ..., "exchange": ...}
                matches = re.findall(r'\{"symbol":[^}]+\}', line)
                for m in matches:
                    try:
                        obj = json.loads(m)
                        if "group" in obj and "date" in obj:
                            telegram_records.append(obj)
                    except Exception:
                        pass

    # Deduplicate while preserving order
    pd_unique = []
    seen_pd = set()
    for r in pd_records:
        key = (r.get("channel_id"), r.get("session_id"), r.get("coin"), r.get("timestamp"))
        if key not in seen_pd:
            seen_pd.add(key)
            pd_unique.append(r)

    tele_unique = []
    seen_tele = set()
    for r in telegram_records:
        key = (r.get("symbol"), r.get("group"), r.get("date"), r.get("hour"), r.get("exchange"))
        if key not in seen_tele:
            seen_tele.add(key)
            tele_unique.append(r)

    print(f"Extracted {len(pd_unique)} unique PD_logs records (expected ~1335).")
    print(f"Extracted {len(tele_unique)} unique Telegram pump records (expected ~1110).")

    # Save PD logs
    pd_out = LABELS_DIR / "PD_logs_cleaned.json"
    with open(pd_out, "w", encoding="utf-8") as f:
        json.dump(pd_unique, f, indent=2)
    print(f"Saved PD logs to {pd_out.resolve()}")

    # Save Telegram pump records
    tele_out = LABELS_DIR / "pump_telegram.json"
    with open(tele_out, "w", encoding="utf-8") as f:
        json.dump(tele_unique, f, indent=2)
    print(f"Saved Pump Telegram records to {tele_out.resolve()}")

    # Also save as CSV for inspection
    df_pd = pd.DataFrame(pd_unique)
    df_pd.to_csv(LABELS_DIR / "PD_logs_cleaned.csv", index=False)
    
    df_tele = pd.DataFrame(tele_unique)
    df_tele.to_csv(LABELS_DIR / "pump_telegram.csv", index=False)

if __name__ == "__main__":
    extract()
