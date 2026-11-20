"""
data_loader.py — Load and validate OHLCV market data and pump-event labels.

Expected CSV schemas
--------------------
OHLCV  (data/raw/ohlcv.csv):
    timestamp   — ISO-8601 or Unix-epoch datetime
    symbol      — e.g. BTC/USDT
    open        — float
    high        — float
    low         — float
    close       — float
    volume      — float

Pump-event labels (data/labels/pump_events.csv):
    symbol          — e.g. BTC/USDT
    event_timestamp — ISO-8601 or Unix-epoch datetime of the known pump event
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Configuration helper                                                #
# ------------------------------------------------------------------ #

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration and return as a dictionary."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path.resolve()}. "
            "Please create config.yaml in the project root."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", config_path.resolve())
    return cfg


# ------------------------------------------------------------------ #
#  OHLCV loader                                                        #
# ------------------------------------------------------------------ #

REQUIRED_OHLCV_COLS = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}


def load_ohlcv(filepath: str) -> pd.DataFrame:
    """
    Load OHLCV CSV, validate columns, parse timestamps, and sort
    chronologically.

    Parameters
    ----------
    filepath : str
        Path to the OHLCV CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV dataframe with a proper ``datetime64`` timestamp column.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"OHLCV data file not found: {filepath.resolve()}\n"
            "Please place your OHLCV CSV at the path specified in config.yaml.\n"
            "Expected columns: timestamp, symbol, open, high, low, close, volume"
        )

    logger.info("Loading OHLCV data from %s …", filepath)
    df = pd.read_csv(filepath)

    # --- Normalize column names ---
    df.columns = df.columns.str.strip().str.lower()

    # --- Validate required columns ---
    missing = REQUIRED_OHLCV_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"OHLCV CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # --- Parse timestamp ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")

    # --- Cast numeric columns ---
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Drop rows where any OHLCV value is NaN (after coercion) ---
    n_before = len(df)
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with non-numeric OHLCV values.", n_dropped)

    # --- Sort chronologically per symbol ---
    df.sort_values(["symbol", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "OHLCV loaded: %d rows, %d symbols, date range %s → %s",
        len(df),
        df["symbol"].nunique(),
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    return df


# ------------------------------------------------------------------ #
#  Pump-event label loader                                             #
# ------------------------------------------------------------------ #

REQUIRED_LABEL_COLS = {"symbol", "event_timestamp"}


def load_pump_labels(filepath: str) -> pd.DataFrame:
    """
    Load pump-event labels from CSV or JSON format.

    Supports both basic schema (symbol, event_timestamp) and rich schema
    (symbol, coin, pair, event_timestamp, exchange, channel_id, group, source_dataset).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        # Fall back to alternative labels file if main doesn't exist
        alt_path = filepath.parent / "pump_events.csv"
        if alt_path.exists():
            filepath = alt_path
        else:
            raise FileNotFoundError(
                f"Pump-event label file not found: {filepath.resolve()}\n"
                "Please place your pump-event labels CSV at the path specified in config.yaml.\n"
                "Expected columns: symbol, event_timestamp"
            )

    logger.info("Loading pump-event labels from %s …", filepath)
    if filepath.suffix == ".json":
        df = pd.read_json(filepath)
    else:
        df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip().str.lower()

    if "event_timestamp" not in df.columns and "timestamp" in df.columns:
        df.rename(columns={"timestamp": "event_timestamp"}, inplace=True)

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True, format="mixed")
    df.sort_values(["symbol", "event_timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "Pump labels loaded: %d events across %d symbols, range %s → %s",
        len(df),
        df["symbol"].nunique(),
        df["event_timestamp"].min(),
        df["event_timestamp"].max(),
    )
    return df


# ------------------------------------------------------------------ #
#  Convenience: load everything                                        #
# ------------------------------------------------------------------ #

def load_all_data(
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both OHLCV data and pump-event labels using paths from *cfg*.

    Parameters
    ----------
    cfg : dict
        Parsed config.yaml dictionary.

    Returns
    -------
    (ohlcv_df, pump_labels_df) : tuple of DataFrames
    """
    ohlcv_path = cfg["data"]["ohlcv_path"]
    labels_path = cfg["data"]["pump_labels_path"]

    ohlcv_df = load_ohlcv(ohlcv_path)
    pump_labels_df = load_pump_labels(labels_path)

    # Quick sanity check — warn if label symbols are not in OHLCV data
    ohlcv_symbols = set(ohlcv_df["symbol"].unique())
    label_symbols = set(pump_labels_df["symbol"].unique())
    unknown = label_symbols - ohlcv_symbols
    if unknown:
        logger.warning(
            "Pump-label file contains symbols not found in OHLCV data: %s. "
            "These labels will be ignored during labeling.",
            unknown,
        )

    return ohlcv_df, pump_labels_df


# ------------------------------------------------------------------ #
#  CLI quick-test                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = load_config()
    try:
        ohlcv, labels = load_all_data(cfg)
        print("\n--- OHLCV Head ---")
        print(ohlcv.head())
        print("\n--- Pump Labels Head ---")
        print(labels.head())
    except FileNotFoundError as exc:
        print(f"\n[WARNING] {exc}")
        print(
            "\nTo get started, place your dataset CSVs at the paths shown above.\n"
            "See README.md for expected formats."
        )
