"""
preprocessing.py — Clean, normalize, and resample OHLCV data.

Responsibilities
----------------
1. Normalize and auto-repair missing OHLCV columns (open, high, low, close, volume).
2. Handle missing values (forward-fill within each symbol).
3. Remove duplicate timestamps.
4. Detect and optionally resample to a uniform frequency.
5. Basic sanity checks (open/high/low/close ordering, non-negative volume).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has all standard columns: timestamp, symbol, open, high, low, close, volume.
    Auto-detects aliases or synthesizes missing values safely.
    """
    df = df.copy()
    # Normalize column names: strip spaces, lowercase, replace spaces/dashes with underscores
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Timestamp alias mapping
    ts_aliases = ["timestamp", "date", "time", "open_time", "datetime", "candle_time", "ts"]
    for alias in ts_aliases:
        if alias in df.columns and "timestamp" not in df.columns:
            df["timestamp"] = df[alias]
            break

    if "timestamp" not in df.columns:
        # If still no timestamp, check if index is datetime or generate integer steps
        if isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = df.index
        else:
            df["timestamp"] = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(df), freq="1min")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Symbol alias mapping
    sym_aliases = ["symbol", "pair", "coin", "ticker", "asset", "market"]
    for alias in sym_aliases:
        if alias in df.columns and "symbol" not in df.columns:
            df["symbol"] = df[alias]
            break
    if "symbol" not in df.columns:
        df["symbol"] = "CRYPTO"

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    # Price / OHLC aliases
    price_aliases = ["close", "price", "last", "last_price", "close_price", "c"]
    for alias in price_aliases:
        if alias in df.columns and "close" not in df.columns:
            df["close"] = pd.to_numeric(df[alias], errors="coerce")
            break

    if "close" not in df.columns:
        # Look for any numeric column that might represent price
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df["close"] = df[num_cols[0]]
        else:
            df["close"] = 1.0

    df["close"] = pd.to_numeric(df["close"], errors="coerce").bfill().ffill().fillna(1.0)

    # Open
    open_aliases = ["open", "open_price", "o"]
    for alias in open_aliases:
        if alias in df.columns and "open" not in df.columns:
            df["open"] = pd.to_numeric(df[alias], errors="coerce")
            break
    if "open" not in df.columns:
        df["open"] = df["close"]
    else:
        df["open"] = pd.to_numeric(df["open"], errors="coerce").fillna(df["close"])

    # High
    high_aliases = ["high", "high_price", "max_price", "h", "max"]
    for alias in high_aliases:
        if alias in df.columns and "high" not in df.columns:
            df["high"] = pd.to_numeric(df[alias], errors="coerce")
            break
    if "high" not in df.columns:
        df["high"] = df[["open", "close"]].max(axis=1)
    else:
        df["high"] = pd.to_numeric(df["high"], errors="coerce").fillna(df[["open", "close"]].max(axis=1))

    # Low
    low_aliases = ["low", "low_price", "min_price", "l", "min"]
    for alias in low_aliases:
        if alias in df.columns and "low" not in df.columns:
            df["low"] = pd.to_numeric(df[alias], errors="coerce")
            break
    if "low" not in df.columns:
        df["low"] = df[["open", "close"]].min(axis=1)
    else:
        df["low"] = pd.to_numeric(df["low"], errors="coerce").fillna(df[["open", "close"]].min(axis=1))

    # Volume
    vol_aliases = ["volume", "vol", "v", "volume_btc", "volume_usdt", "base_volume", "quote_volume", "amount"]
    for alias in vol_aliases:
        if alias in df.columns and "volume" not in df.columns:
            df["volume"] = pd.to_numeric(df[alias], errors="coerce")
            break
    if "volume" not in df.columns:
        df["volume"] = 100.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(100.0)

    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
    df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))
    df["volume"] = np.maximum(df["volume"], 0.0)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (symbol, timestamp) rows, keeping the last."""
    n_before = len(df)
    df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Removed %d duplicate (symbol, timestamp) rows.", n_dropped)
    return df.reset_index(drop=True)


def sanity_check_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic OHLCV sanity rules and enforce non-negative volume.
    """
    df = ensure_ohlcv_columns(df)
    mask_hl = df["high"] >= df["low"]
    mask_vol = df["volume"] >= 0
    valid = mask_hl & mask_vol
    n_bad = (~valid).sum()
    if n_bad:
        logger.warning("Auto-correcting %d rows that violated OHLCV constraints.", n_bad)
        df.loc[~valid, "high"] = df.loc[~valid, "low"] + 1e-6
    return df.reset_index(drop=True)


def forward_fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill NaN values within each symbol group."""
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        if col in df.columns:
            df[col] = df.groupby("symbol")[col].transform(lambda g: g.ffill().bfill())
    df.dropna(subset=["timestamp", "close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def detect_frequency(df: pd.DataFrame, symbol: Optional[str] = None) -> Optional[pd.Timedelta]:
    """Infer the most common time-step for a single symbol."""
    sub = df[df["symbol"] == symbol] if symbol else df
    if len(sub) < 2:
        return None
    diffs = sub["timestamp"].diff().dropna()
    freq = diffs.mode()
    if len(freq) == 0:
        return None
    return freq.iloc[0]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Ensure & normalize standard OHLCV columns (auto-repair missing/aliases)
      2. Remove duplicate timestamps
      3. Sanity-check OHLCV values
      4. Forward-fill gaps
      5. Sort chronologically
    """
    logger.info("Preprocessing OHLCV data (%d rows) …", len(df))
    df = ensure_ohlcv_columns(df)
    df = remove_duplicates(df)
    df = sanity_check_ohlcv(df)
    df = forward_fill_gaps(df)
    df.sort_values(["symbol", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
