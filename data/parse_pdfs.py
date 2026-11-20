import json
import re
from pathlib import Path
import pypdf
import pandas as pd

user_uploaded_dir = Path(r"C:\Users\shrey\.gemini\antigravity\brain\24528497-c421-487d-9a34-cccafab930b9\.user_uploaded")
pdf_files = sorted(list(user_uploaded_dir.glob("*.pdf")))

pd_records = []
telegram_records = []

json_block_pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)

def clean_ts(ts: str) -> str:
    if not isinstance(ts, str):
        return ts
    ts = ts.strip()
    # Fix broken year prefix from PDF line break e.g. '020-06-03' -> '2020-06-03'
    ts = re.sub(r"\b0(1\d|2\d)-", r"20\1-", ts)
    # Fix spaces within date
    match = re.search(r"(\d{2,4}[\s\d-]*\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}.*)", ts)
    if match:
        date_str = re.sub(r"\s+", "", match.group(1))
        date_str = re.sub(r"\b0(1\d|2\d)-", r"20\1-", date_str)
        time_str = match.group(2)
        return f"{date_str} {time_str}"
    return re.sub(r"\s+", " ", ts)

for pdf_path in pdf_files:
    reader = pypdf.PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    print(f"\nProcessing {pdf_path.name} ({num_pages} pages)...")
    
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"
        
    matches = json_block_pattern.findall(full_text)
    
    for m in matches:
        cleaned_json = re.sub(r"\s+", " ", m).strip()
        try:
            obj = json.loads(cleaned_json)
            if "timestamp" in obj:
                obj["timestamp"] = clean_ts(obj["timestamp"])
            if "channel_id" in obj:
                pd_records.append(obj)
            elif "group" in obj or "symbol" in obj:
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

print(f"\nExtracted {len(pd_unique)} unique PD logs records (Expected: 1335).")
print(f"Extracted {len(tele_unique)} unique Telegram pump records (Expected: 1110).")

labels_dir = Path("data/labels")
labels_dir.mkdir(parents=True, exist_ok=True)

if pd_unique:
    df_pd = pd.DataFrame(pd_unique)
    df_pd.to_json(labels_dir / "PD_logs_cleaned.json", orient="records", indent=2)
    df_pd.to_csv(labels_dir / "PD_logs_cleaned.csv", index=False)
    print(f"Saved {len(df_pd)} PD logs records to data/labels/PD_logs_cleaned.json and CSV")

if tele_unique:
    df_tele = pd.DataFrame(tele_unique)
    df_tele.to_json(labels_dir / "pump_telegram.json", orient="records", indent=2)
    df_tele.to_csv(labels_dir / "pump_telegram.csv", index=False)
    print(f"Saved {len(df_tele)} Telegram records to data/labels/pump_telegram.json and CSV")
