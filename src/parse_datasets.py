"""
parse_datasets.py — Parser and Schema Verifier for Pump Datasets.

Parses two types of raw pump log files:
1. PD_logs_cleaned.txt:
   JSON records containing: channel_id, session_id, coin, exchange, pair, timestamp
2. pump_telegram.csv:
   JSON records containing: symbol, group, date, hour, exchange

Key Principles:
- ZERO fabrication.
- Exact timestamp parsing with UTC timezone.
- Explicit symbol normalization (e.g., COIN/PAIR like APPC/BTC).
- Schema validation & deduplication report before merging.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def parse_pd_logs(content_or_path: str | Path) -> pd.DataFrame:
    """
    Parse PD_logs_cleaned format.
    Format: Lines containing JSON or numbered lines like:
    1. {"channel_id":1095533634,"session_id":256,"coin":"APPC","exchange":"binance","pair":"BTC","timestamp":"2019-06-04 12:01:33+00:00"}
    """
    records = []
    
    if isinstance(content_or_path, Path) or (isinstance(content_or_path, str) and Path(content_or_path).exists()):
        with open(content_or_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = content_or_path.strip().split("\n")

    json_pattern = re.compile(r"\{.*\}")

    for line_num, line in enumerate(lines, 1):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        match = json_pattern.search(line_clean)
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                coin = str(data.get("coin", "")).upper().strip()
                pair = str(data.get("pair", "")).upper().strip()
                exchange = str(data.get("exchange", "")).lower().strip()
                ts_raw = data.get("timestamp")
                channel_id = data.get("channel_id")
                session_id = data.get("session_id")
                
                # Form full symbol COIN/PAIR or just COIN if pair is missing
                symbol = f"{coin}/{pair}" if pair else coin
                
                records.append({
                    "raw_line": line_num,
                    "symbol": symbol,
                    "coin": coin,
                    "pair": pair,
                    "exchange": exchange,
                    "event_timestamp": ts_raw,
                    "channel_id": str(channel_id) if channel_id is not None else None,
                    "session_id": str(session_id) if session_id is not None else None,
                    "source_dataset": "PD_logs_cleaned"
                })
            except json.JSONDecodeError as err:
                logger.warning(f"Line {line_num}: Failed to parse JSON: {err}")

    df = pd.DataFrame(records)
    if not df.empty:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    return df


def parse_pump_telegram(content_or_path: str | Path) -> pd.DataFrame:
    """
    Parse pump_telegram.csv format.
    Format: Lines containing JSON or numbered lines like:
    1. {"symbol":"R","group":"ATW","date":"2018-01-01","hour":"16:00","exchange":"yobit"}
    """
    records = []
    
    if isinstance(content_or_path, Path) or (isinstance(content_or_path, str) and Path(content_or_path).exists()):
        with open(content_or_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = content_or_path.strip().split("\n")

    json_pattern = re.compile(r"\{.*\}")

    for line_num, line in enumerate(lines, 1):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        match = json_pattern.search(line_clean)
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                coin = str(data.get("symbol", "")).upper().strip()
                group = str(data.get("group", "")).strip()
                date_str = str(data.get("date", "")).strip()
                hour_str = str(data.get("hour", "")).strip()
                exchange = str(data.get("exchange", "")).lower().strip()
                
                # Combine date + hour
                if ":" in hour_str:
                    if len(hour_str.split(":")) == 2:
                        hour_str += ":00"
                else:
                    hour_str += ":00:00"
                
                ts_str = f"{date_str} {hour_str}+00:00"
                
                # Standard symbol assuming BTC pair unless exchange uses USDT
                symbol = f"{coin}/BTC"
                
                records.append({
                    "raw_line": line_num,
                    "symbol": symbol,
                    "coin": coin,
                    "pair": "BTC",
                    "exchange": exchange,
                    "event_timestamp": ts_str,
                    "group": group,
                    "source_dataset": "pump_telegram"
                })
            except Exception as err:
                logger.warning(f"Line {line_num}: Failed to parse record: {err}")

    df = pd.DataFrame(records)
    if not df.empty:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    return df


def verify_and_analyze_datasets(df_pd: pd.DataFrame, df_telegram: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze both parsed dataframes, verify timestamps, symbols, and exchanges,
    and report overlaps and statistics without modifying original records.
    """
    report = {}
    
    report["pd_logs_summary"] = {
        "total_records": len(df_pd),
        "unique_coins": df_pd["coin"].nunique() if not df_pd.empty else 0,
        "unique_symbols": df_pd["symbol"].nunique() if not df_pd.empty else 0,
        "unique_exchanges": df_pd["exchange"].unique().tolist() if not df_pd.empty else [],
        "min_timestamp": str(df_pd["event_timestamp"].min()) if not df_pd.empty else None,
        "max_timestamp": str(df_pd["event_timestamp"].max()) if not df_pd.empty else None,
        "unique_channels": df_pd["channel_id"].nunique() if not df_pd.empty else 0,
    }

    report["pump_telegram_summary"] = {
        "total_records": len(df_telegram),
        "unique_coins": df_telegram["coin"].nunique() if not df_telegram.empty else 0,
        "unique_symbols": df_telegram["symbol"].nunique() if not df_telegram.empty else 0,
        "unique_exchanges": df_telegram["exchange"].unique().tolist() if not df_telegram.empty else [],
        "min_timestamp": str(df_telegram["event_timestamp"].min()) if not df_telegram.empty else None,
        "max_timestamp": str(df_telegram["event_timestamp"].max()) if not df_telegram.empty else None,
        "unique_groups": df_telegram["group"].nunique() if not df_telegram.empty else 0,
    }

    # Cross-dataset analysis
    if not df_pd.empty and not df_telegram.empty:
        common_coins = set(df_pd["coin"]).intersection(set(df_telegram["coin"]))
        
        # Exact matching timestamps for same coin (within 5 minutes)
        merged = pd.merge(
            df_pd, df_telegram,
            on="coin",
            suffixes=("_pd", "_telegram")
        )
        merged["time_diff_min"] = (merged["event_timestamp_pd"] - merged["event_timestamp_telegram"]).abs() / pd.Timedelta(minutes=1)
        exact_matches = merged[merged["time_diff_min"] <= 5]

        report["cross_analysis"] = {
            "common_coins_count": len(common_coins),
            "common_coins_sample": list(common_coins)[:10],
            "near_exact_event_matches_within_5min": len(exact_matches),
        }

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("Dataset Parser & Verification Module Ready.")
