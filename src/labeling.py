"""
labeling.py — Create future-looking binary pump/dump target labels.

Labeling Logic (NO future data leakage)
----------------------------------------
For each observation at time ``t`` for a given ``symbol``:

    target = 1   if there exists at least one known pump event for that
                 symbol in the interval  (t, t + prediction_horizon]

    target = 0   otherwise

This means the label looks *forward* from the observation timestamp.
Features computed at time ``t`` must only use data from time <= ``t`` to
avoid leakage — that contract is enforced by the feature-engineering module,
not here.

Example
-------
If ``prediction_horizon = 30 minutes`` and a pump event occurs at 12:00:

    - Observation at 11:25  →  pump at 12:00 is 35 min away  →  target = 0
    - Observation at 11:30  →  pump at 12:00 is 30 min away  →  target = 1
    - Observation at 11:45  →  pump at 12:00 is 15 min away  →  target = 1
    - Observation at 11:59  →  pump at 12:00 is  1 min away  →  target = 1
    - Observation at 12:00  →  pump at 12:00 is  0 min away  →  target = 0
      (we predict *before* the event; the event itself is not a prediction)
    - Observation at 12:05  →  pump at 12:00 is in the past  →  target = 0
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_binary_labels(
    ohlcv: pd.DataFrame,
    pump_events: pd.DataFrame,
    prediction_horizon_minutes: int = 30,
) -> pd.DataFrame:
    """
    Attach a binary ``target`` column to the OHLCV dataframe.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Preprocessed OHLCV data with columns [timestamp, symbol, ...].
        Must be sorted by (symbol, timestamp).
    pump_events : pd.DataFrame
        Known pump events with columns [symbol, event_timestamp].
    prediction_horizon_minutes : int
        How many minutes into the future to look for a pump event.

    Returns
    -------
    pd.DataFrame
        Copy of ``ohlcv`` with an additional ``target`` column (0 or 1)
        and a ``time_to_next_pump_min`` column (float, NaN if no upcoming
        pump exists).
    """
    horizon = pd.Timedelta(minutes=prediction_horizon_minutes)
    logger.info(
        "Creating binary labels with prediction horizon = %d minutes …",
        prediction_horizon_minutes,
    )

    df = ohlcv.copy()
    df["target"] = 0
    df["time_to_next_pump_min"] = np.nan

    symbols_with_events = set(pump_events["symbol"].unique())

    total_positive = 0

    for symbol in df["symbol"].unique():
        mask_sym = df["symbol"] == symbol

        if symbol not in symbols_with_events:
            # No pump events for this symbol — all targets stay 0
            continue

        sym_events = pump_events.loc[
            pump_events["symbol"] == symbol, "event_timestamp"
        ].sort_values().values  # numpy datetime64 array

        sym_idx = df.index[mask_sym]
        timestamps = df.loc[sym_idx, "timestamp"].values  # datetime64 array

        # For each observation, find the time to the *next* pump event
        # using searchsorted for efficiency.
        insert_pos = np.searchsorted(sym_events, timestamps, side="right")

        for i, (idx, ts, pos) in enumerate(zip(sym_idx, timestamps, insert_pos)):
            if pos < len(sym_events):
                next_event = sym_events[pos]
                delta = next_event - ts
                delta_minutes = delta / np.timedelta64(1, "m")

                df.at[idx, "time_to_next_pump_min"] = delta_minutes

                # Strictly *before* the event and within the horizon
                if 0 < delta_minutes <= prediction_horizon_minutes:
                    df.at[idx, "target"] = 1
                    total_positive += 1

    n_total = len(df)
    n_pos = total_positive
    n_neg = n_total - n_pos
    pct = 100.0 * n_pos / n_total if n_total else 0.0

    logger.info(
        "Labeling complete: %d total, %d positive (%.2f%%), %d negative",
        n_total, n_pos, pct, n_neg,
    )

    if n_pos == 0:
        logger.warning(
            "No positive labels were created! Check that pump_events timestamps "
            "fall within the OHLCV data range and the prediction horizon is "
            "large enough."
        )

    return df


def label_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-symbol label distribution statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled dataframe (must contain ``symbol`` and ``target`` columns).

    Returns
    -------
    pd.DataFrame
        Summary with columns [symbol, total, positive, negative, pct_positive].
    """
    stats = (
        df.groupby("symbol")["target"]
        .agg(total="count", positive="sum")
        .reset_index()
    )
    stats["negative"] = stats["total"] - stats["positive"]
    stats["pct_positive"] = 100.0 * stats["positive"] / stats["total"]
    return stats


# ------------------------------------------------------------------ #
#  CLI quick-test                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_loader import load_config, load_all_data
    from src.preprocessing import preprocess

    cfg = load_config()
    try:
        ohlcv, labels = load_all_data(cfg)
        ohlcv = preprocess(ohlcv)

        horizon = cfg.get("prediction_horizon_minutes", 30)
        labeled = create_binary_labels(ohlcv, labels, horizon)

        print("\n--- Label Distribution ---")
        print(label_statistics(labeled).to_string(index=False))
        print(f"\nHorizon: {horizon} min")
        print(labeled[["timestamp", "symbol", "close", "volume", "target", "time_to_next_pump_min"]].head(20))
    except FileNotFoundError as exc:
        print(f"\n[WARNING] {exc}")
