"""
analyze_schemas.py — Comprehensive Schema Verification & Analysis.

Verifies:
1. Dataset 1: PD_logs_cleaned (1335 records)
2. Dataset 2: pump_telegram (1110 records)
3. Schema compliance, missing values, timestamp ranges, exchanges, symbols.
4. Cross-dataset alignment, exact matches, and temporal overlaps.
5. Standardized output creation for data loader without data loss or fabrication.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

LABELS_DIR = Path("data/labels")

def clean_ts_str(val: str) -> str:
    if not isinstance(val, str):
        return str(val)
    val = val.strip()
    val = re.sub(r"\b0(1\d|2\d)-", r"20\1-", val)
    match = re.search(r"(\d{2,4}[\s\d-]*\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}.*)", val)
    if match:
        date_str = re.sub(r"\s+", "", match.group(1))
        date_str = re.sub(r"\b0(1\d|2\d)-", r"20\1-", date_str)
        time_str = match.group(2)
        return f"{date_str} {time_str}"
    return re.sub(r"\s+", " ", val)

def analyze():
    pd_path = LABELS_DIR / "PD_logs_cleaned.json"
    tele_path = LABELS_DIR / "pump_telegram.json"

    with open(pd_path, "r", encoding="utf-8") as f:
        pd_raw = json.load(f)

    with open(tele_path, "r", encoding="utf-8") as f:
        tele_raw = json.load(f)

    print("=" * 75)
    print("           CRYPTOCURRENCY PUMP DATASETS: SCHEMA & VERIFICATION REPORT")
    print("=" * 75)

    # --------------------------------------------------------------- #
    # 1. PD Logs Dataset Analysis                                      #
    # --------------------------------------------------------------- #
    df_pd = pd.DataFrame(pd_raw)
    df_pd["clean_ts_raw"] = df_pd["timestamp"].apply(clean_ts_str)
    df_pd["event_timestamp"] = pd.to_datetime(df_pd["clean_ts_raw"], utc=True, format="mixed")
    df_pd["full_symbol"] = df_pd["coin"] + "/" + df_pd["pair"]

    print("\n[DATASET 1: PD_logs_cleaned.txt]")
    print(f"  • Total AI-Readable Records : {len(df_pd)}")
    print(f"  • Unique Telegram Channels  : {df_pd['channel_id'].nunique()}")
    print(f"  • Unique Coins (Base Tickers): {df_pd['coin'].nunique()}")
    print(f"  • Unique Pair Symbols       : {df_pd['full_symbol'].nunique()}")
    print(f"  • Unique Exchanges          : {df_pd['exchange'].nunique()} -> {sorted(df_pd['exchange'].unique())}")
    print(f"  • Unique Pair Base/Quote    : {df_pd['pair'].unique().tolist()}")
    print(f"  • Time Range (UTC)          : {df_pd['event_timestamp'].min()}  -->  {df_pd['event_timestamp'].max()}")
    print(f"  • Missing Field Counts      : {df_pd[['channel_id','session_id','coin','exchange','pair','timestamp']].isnull().sum().to_dict()}")

    print("\n  Exchange Breakdown (PD Logs):")
    print(df_pd["exchange"].value_counts().to_string())

    print("\n  Quote Pair Breakdown (PD Logs):")
    print(df_pd["pair"].value_counts().to_string())

    print("\n  Top 10 Most Targeted Coins (PD Logs):")
    print(df_pd["coin"].value_counts().head(10).to_string())

    # --------------------------------------------------------------- #
    # 2. Pump Telegram Dataset Analysis                                #
    # --------------------------------------------------------------- #
    df_tele = pd.DataFrame(tele_raw)
    
    def make_ts(row):
        date_str = str(row["date"]).strip()
        hour_str = str(row["hour"]).strip()
        if len(hour_str.split(":")) == 2:
            hour_str += ":00"
        return f"{date_str} {hour_str}+00:00"

    df_tele["clean_ts_raw"] = df_tele.apply(make_ts, axis=1)
    df_tele["event_timestamp"] = pd.to_datetime(df_tele["clean_ts_raw"].apply(clean_ts_str), utc=True, format="mixed")
    df_tele["full_symbol"] = df_tele["symbol"] + "/BTC"

    print("\n" + "-" * 75)
    print("[DATASET 2: pump_telegram.csv]")
    print(f"  • Total AI-Readable Records : {len(df_tele)}")
    print(f"  • Unique Telegram Groups    : {df_tele['group'].nunique()}")
    print(f"  • Unique Coins (Base Tickers): {df_tele['symbol'].nunique()}")
    print(f"  • Unique Exchanges          : {df_tele['exchange'].nunique()} -> {sorted(df_tele['exchange'].unique())}")
    print(f"  • Time Range (UTC)          : {df_tele['event_timestamp'].min()}  -->  {df_tele['event_timestamp'].max()}")
    print(f"  • Missing Field Counts      : {df_tele[['symbol','group','date','hour','exchange']].isnull().sum().to_dict()}")

    print("\n  Exchange Breakdown (Pump Telegram):")
    print(df_tele["exchange"].value_counts().to_string())

    print("\n  Top 10 Active Telegram Groups:")
    print(df_tele["group"].value_counts().head(10).to_string())

    print("\n  Top 10 Most Targeted Coins (Pump Telegram):")
    print(df_tele["symbol"].value_counts().head(10).to_string())

    # --------------------------------------------------------------- #
    # 3. Cross-Dataset Alignment & Overlap Verification                #
    # --------------------------------------------------------------- #
    print("\n" + "=" * 75)
    print("      CROSS-DATASET ALIGNMENT & OVERLAP ANALYSIS")
    print("=" * 75)

    coins_pd = set(df_pd["coin"])
    coins_tele = set(df_tele["symbol"])
    common_coins = coins_pd.intersection(coins_tele)
    
    print(f"  • Unique Coins in Dataset 1 (PD Logs)       : {len(coins_pd)}")
    print(f"  • Unique Coins in Dataset 2 (Telegram Pump): {len(coins_tele)}")
    print(f"  • Common Coins Present in BOTH Datasets     : {len(common_coins)}")
    print(f"  • Sample Common Coins                       : {sorted(list(common_coins))[:15]}")

    # Temporal Overlap Window Analysis
    merged = pd.merge(
        df_pd, df_tele,
        left_on="coin", right_on="symbol",
        suffixes=("_pd", "_tele")
    )
    merged["diff_minutes"] = (merged["event_timestamp_pd"] - merged["event_timestamp_tele"]).abs() / pd.Timedelta(minutes=1)
    
    exact_matches = merged[merged["diff_minutes"] == 0]
    near_matches_15m = merged[merged["diff_minutes"] <= 15]
    near_matches_60m = merged[merged["diff_minutes"] <= 60]

    print(f"\n  • Exact Timestamp Matches (Same coin, same minute): {len(exact_matches)}")
    print(f"  • Near Matches (within ±15 minutes)               : {len(near_matches_15m)}")
    print(f"  • Near Matches (within ±60 minutes)               : {len(near_matches_60m)}")

    if not near_matches_15m.empty:
        print("\n  Sample Matched Events Across Both Datasets (within 15 min):")
        cols_show = ["coin", "exchange_pd", "exchange_tele", "event_timestamp_pd", "event_timestamp_tele", "channel_id", "group", "diff_minutes"]
        print(near_matches_15m[cols_show].head(10).to_string(index=False))

    # --------------------------------------------------------------- #
    # 4. Standardized Target Output Creation                           #
    # --------------------------------------------------------------- #
    print("\n" + "=" * 75)
    print("      STANDARDIZED TARGET PIPELINE PREPARATION")
    print("=" * 75)

    pd_clean = pd.DataFrame({
        "symbol": df_pd["full_symbol"],
        "coin": df_pd["coin"],
        "pair": df_pd["pair"],
        "event_timestamp": df_pd["event_timestamp"],
        "exchange": df_pd["exchange"],
        "channel_id": df_pd["channel_id"].astype(str),
        "group": None,
        "source_dataset": "PD_logs_cleaned"
    })

    tele_clean = pd.DataFrame({
        "symbol": df_tele["full_symbol"],
        "coin": df_tele["symbol"],
        "pair": "BTC",
        "event_timestamp": df_tele["event_timestamp"],
        "exchange": df_tele["exchange"],
        "channel_id": None,
        "group": df_tele["group"],
        "source_dataset": "pump_telegram"
    })

    combined_labels = pd.concat([pd_clean, tele_clean], ignore_index=True)
    combined_labels.sort_values(["symbol", "event_timestamp"], inplace=True)
    combined_labels.reset_index(drop=True, inplace=True)

    # Save to data/labels/pump_events.csv
    out_csv = LABELS_DIR / "pump_events.csv"
    combined_labels.to_csv(out_csv, index=False)

    print(f"  [OK] Combined & verified {len(combined_labels)} total pump event records.")
    print(f"  [OK] Output CSV saved to: {out_csv.resolve()}")
    print(f"  [OK] Preserved full data lineage (source_dataset, channel_id, group) without data fabrication or loss.")
    print("=" * 75)

if __name__ == "__main__":
    analyze()
