import json
import re
import sys
from pathlib import Path
import pandas as pd

p = Path(r"C:\Users\shrey\.gemini\antigravity\brain\24528497-c421-487d-9a34-cccafab930b9\.system_generated\logs\transcript_full.jsonl")

pd_records = []
telegram_records = []

with open(p, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        raw_text = json.dumps(data)

        # PD Logs: {"channel_id": ..., "session_id": ..., "coin": ..., "exchange": ..., "pair": ..., "timestamp": ...}
        for match in re.finditer(r'\{"channel_id":\s*\d+,"session_id":[^}]+\}', raw_text):
            try:
                obj = json.loads(match.group(0))
                pd_records.append(obj)
            except Exception:
                pass

        # Pump Telegram: {"symbol": ..., "group": ..., "date": ..., "hour": ..., "exchange": ...}
        for match in re.finditer(r'\{"symbol":[^}]+"group":[^}]+\}', raw_text):
            try:
                obj = json.loads(match.group(0))
                telegram_records.append(obj)
            except Exception:
                pass

# Deduplicate
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

print(f"Parsed {len(pd_unique)} PD logs records.")
print(f"Parsed {len(tele_unique)} Telegram pump records.")

labels_dir = Path("data/labels")
labels_dir.mkdir(parents=True, exist_ok=True)

with open(labels_dir / "PD_logs_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(pd_unique, f, indent=2)

with open(labels_dir / "pump_telegram.json", "w", encoding="utf-8") as f:
    json.dump(tele_unique, f, indent=2)

pd.DataFrame(pd_unique).to_csv(labels_dir / "PD_logs_cleaned.csv", index=False)
pd.DataFrame(tele_unique).to_csv(labels_dir / "pump_telegram.csv", index=False)

print("Saved clean dataset files into data/labels/")
