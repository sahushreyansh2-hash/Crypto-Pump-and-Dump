"""
split.py — Chronological train / validation / test splitting.

IMPORTANT: We NEVER use random splitting for time-series data.

The data is sorted chronologically and split so that:
    - Training set   : first 70% of data (earliest)
    - Validation set  : next 15%
    - Test set        : last 15% (latest)

Split is performed **per-symbol** so each symbol's temporal ordering is
preserved. The split boundaries (timestamps) are logged and saved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe chronologically within each symbol.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``symbol`` and ``timestamp`` columns and be sorted
        by (symbol, timestamp).
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of DataFrames

    Raises
    ------
    ValueError
        If ratios do not sum to ~1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.4f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )

    train_parts, val_parts, test_parts = [], [], []
    split_info: Dict[str, dict] = {}

    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_chunk = group.iloc[:train_end]
        val_chunk = group.iloc[train_end:val_end]
        test_chunk = group.iloc[val_end:]

        train_parts.append(train_chunk)
        val_parts.append(val_chunk)
        test_parts.append(test_chunk)

        split_info[symbol] = {
            "total_rows": n,
            "train_rows": len(train_chunk),
            "val_rows": len(val_chunk),
            "test_rows": len(test_chunk),
            "train_start": str(train_chunk["timestamp"].iloc[0]) if len(train_chunk) else "N/A",
            "train_end": str(train_chunk["timestamp"].iloc[-1]) if len(train_chunk) else "N/A",
            "val_start": str(val_chunk["timestamp"].iloc[0]) if len(val_chunk) else "N/A",
            "val_end": str(val_chunk["timestamp"].iloc[-1]) if len(val_chunk) else "N/A",
            "test_start": str(test_chunk["timestamp"].iloc[0]) if len(test_chunk) else "N/A",
            "test_end": str(test_chunk["timestamp"].iloc[-1]) if len(test_chunk) else "N/A",
        }

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # --- Logging ---
    logger.info("Chronological split complete:")
    logger.info(
        "  Train : %d rows (%.1f%%)",
        len(train_df), 100.0 * len(train_df) / len(df),
    )
    logger.info(
        "  Val   : %d rows (%.1f%%)",
        len(val_df), 100.0 * len(val_df) / len(df),
    )
    logger.info(
        "  Test  : %d rows (%.1f%%)",
        len(test_df), 100.0 * len(test_df) / len(df),
    )

    for sym, info in split_info.items():
        logger.info(
            "  [%s] train %s → %s | val %s → %s | test %s → %s",
            sym,
            info["train_start"], info["train_end"],
            info["val_start"], info["val_end"],
            info["test_start"], info["test_end"],
        )

    return train_df, val_df, test_df


def split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a summary DataFrame of split statistics including label counts.
    """
    rows = []
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n = len(part)
        n_pos = int(part["target"].sum()) if "target" in part.columns else 0
        n_neg = n - n_pos
        rows.append({
            "split": name,
            "rows": n,
            "positive": n_pos,
            "negative": n_neg,
            "pct_positive": round(100.0 * n_pos / n, 4) if n else 0.0,
            "start_time": str(part["timestamp"].min()) if n else "N/A",
            "end_time": str(part["timestamp"].max()) if n else "N/A",
        })
    return pd.DataFrame(rows)


def save_split_info(
    summary: pd.DataFrame,
    output_dir: str = "results",
) -> None:
    """Save split summary to CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "split_summary.csv"
    summary.to_csv(path, index=False)
    logger.info("Split summary saved to %s", path)


# ------------------------------------------------------------------ #
#  CLI quick-test                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_loader import load_config, load_all_data
    from src.preprocessing import preprocess
    from src.labeling import create_binary_labels
    from src.features import engineer_features

    cfg = load_config()
    try:
        ohlcv, labels = load_all_data(cfg)
        ohlcv = preprocess(ohlcv)
        horizon = cfg.get("prediction_horizon_minutes", 30)
        labeled = create_binary_labels(ohlcv, labels, horizon)
        featured = engineer_features(labeled, cfg)

        train, val, test = chronological_split(
            featured,
            cfg.get("train_ratio", 0.70),
            cfg.get("validation_ratio", 0.15),
            cfg.get("test_ratio", 0.15),
        )

        summary = split_summary(train, val, test)
        print("\n--- Split Summary ---")
        print(summary.to_string(index=False))
        save_split_info(summary, cfg["data"].get("results_dir", "results"))

    except FileNotFoundError as exc:
        print(f"\n[WARNING] {exc}")
