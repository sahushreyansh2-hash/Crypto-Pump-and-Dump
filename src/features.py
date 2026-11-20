"""
features.py — Feature engineering for pump-and-dump early detection.

All rolling / lookback features use ONLY current and historical data.
No future data is used — this is critical to prevent leakage.

Feature Groups
--------------
1. PRICE      — returns, log returns, rolling returns, distance from MAs
2. VOLUME     — change, rolling stats, z-score, abnormal ratio
3. VOLATILITY — rolling std, price range, HL%, ATR-like
4. MOMENTUM   — RSI, momentum, MA relationships
5. TIME       — hour, minute, day-of-week, cyclical encodings
6. PRICE_VOL  — price-volume correlation, volume acceleration, interactions

Each group is computed by a dedicated function and tagged with a
``feature_group`` attribute so the ablation study can select subsets.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Registry: maps feature-group name → list of column names            #
# ------------------------------------------------------------------ #
FEATURE_REGISTRY: Dict[str, List[str]] = {}


def _register(group: str, cols: List[str]) -> None:
    FEATURE_REGISTRY.setdefault(group, []).extend(cols)


# ------------------------------------------------------------------ #
#  Helper: safe rolling (min_periods=window to avoid partial windows)  #
# ------------------------------------------------------------------ #

def _rolling(series: pd.Series, window: int, func: str = "mean") -> pd.Series:
    """Apply rolling function with min_periods=window (no partial data)."""
    r = series.rolling(window=window, min_periods=window)
    return getattr(r, func)()


# ================================================================== #
#  1. PRICE FEATURES                                                   #
# ================================================================== #

def add_price_features(
    df: pd.DataFrame,
    rolling_windows: List[int],
    ma_periods: List[int],
) -> pd.DataFrame:
    """Percentage return, log return, rolling returns, MA distance."""
    cols_added: List[str] = []

    # Percentage return
    df["pct_return"] = df.groupby("symbol")["close"].pct_change()
    cols_added.append("pct_return")

    # Log return
    df["log_return"] = df.groupby("symbol")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    cols_added.append("log_return")

    # Price change (absolute)
    df["price_change"] = df.groupby("symbol")["close"].diff()
    cols_added.append("price_change")

    # Rolling returns
    for w in rolling_windows:
        col = f"rolling_return_{w}"
        df[col] = df.groupby("symbol")["close"].transform(
            lambda s, _w=w: s.pct_change(periods=_w)
        )
        cols_added.append(col)

    # Distance from moving averages
    for p in ma_periods:
        ma_col = f"ma_{p}"
        dist_col = f"dist_ma_{p}"
        df[ma_col] = df.groupby("symbol")["close"].transform(
            lambda s, _p=p: s.rolling(_p, min_periods=_p).mean()
        )
        df[dist_col] = (df["close"] - df[ma_col]) / df[ma_col]
        cols_added.extend([ma_col, dist_col])

    _register("price", cols_added)
    return df


# ================================================================== #
#  2. VOLUME FEATURES                                                  #
# ================================================================== #

def add_volume_features(
    df: pd.DataFrame,
    rolling_windows: List[int],
    zscore_window: int = 20,
) -> pd.DataFrame:
    """Volume change, rolling mean/std, z-score, abnormal ratio."""
    cols_added: List[str] = []

    # Volume change
    df["volume_change"] = df.groupby("symbol")["volume"].pct_change()
    cols_added.append("volume_change")

    for w in rolling_windows:
        mean_col = f"vol_rolling_mean_{w}"
        std_col = f"vol_rolling_std_{w}"
        df[mean_col] = df.groupby("symbol")["volume"].transform(
            lambda s, _w=w: s.rolling(_w, min_periods=_w).mean()
        )
        df[std_col] = df.groupby("symbol")["volume"].transform(
            lambda s, _w=w: s.rolling(_w, min_periods=_w).std()
        )
        cols_added.extend([mean_col, std_col])

    # Volume z-score (using zscore_window)
    df["vol_zscore"] = df.groupby("symbol")["volume"].transform(
        lambda s: (
            (s - s.rolling(zscore_window, min_periods=zscore_window).mean())
            / s.rolling(zscore_window, min_periods=zscore_window).std()
        )
    )
    cols_added.append("vol_zscore")

    # Abnormal volume ratio (current volume / rolling mean)
    mean_ref = f"vol_rolling_mean_{zscore_window}"
    if mean_ref not in df.columns:
        df[mean_ref] = df.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(zscore_window, min_periods=zscore_window).mean()
        )
    df["abnormal_vol_ratio"] = df["volume"] / df[mean_ref].replace(0, np.nan)
    cols_added.append("abnormal_vol_ratio")

    _register("volume", cols_added)
    return df


# ================================================================== #
#  3. VOLATILITY FEATURES                                              #
# ================================================================== #

def add_volatility_features(
    df: pd.DataFrame,
    rolling_windows: List[int],
    atr_period: int = 14,
) -> pd.DataFrame:
    """Rolling volatility, price range, HL%, ATR-like."""
    cols_added: List[str] = []

    for w in rolling_windows:
        vol_col = f"volatility_{w}"
        df[vol_col] = df.groupby("symbol")["log_return"].transform(
            lambda s, _w=w: s.rolling(_w, min_periods=_w).std()
        )
        cols_added.append(vol_col)

    # Intra-bar price range (absolute and percentage)
    df["price_range"] = df["high"] - df["low"]
    df["hl_pct"] = df["price_range"] / df["low"].replace(0, np.nan)
    cols_added.extend(["price_range", "hl_pct"])

    # ATR-like feature (Average True Range)
    df["true_range"] = df.groupby("symbol").apply(
        lambda g: pd.concat([
            g["high"] - g["low"],
            (g["high"] - g["close"].shift(1)).abs(),
            (g["low"] - g["close"].shift(1)).abs(),
        ], axis=1).max(axis=1),
        include_groups=False,
    ).droplevel(0).sort_index()

    df["atr"] = df.groupby("symbol")["true_range"].transform(
        lambda s: s.rolling(atr_period, min_periods=atr_period).mean()
    )
    cols_added.extend(["true_range", "atr"])

    _register("volatility", cols_added)
    return df


# ================================================================== #
#  4. MOMENTUM FEATURES                                                #
# ================================================================== #

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def add_momentum_features(
    df: pd.DataFrame,
    rsi_period: int = 14,
    ma_periods: Optional[List[int]] = None,
) -> pd.DataFrame:
    """RSI, momentum (rate of change), MA cross signals."""
    cols_added: List[str] = []

    # RSI
    df["rsi"] = df.groupby("symbol")["close"].transform(
        lambda s: _rsi(s, rsi_period)
    )
    cols_added.append("rsi")

    # Momentum (n-period rate of change)
    for n in [5, 10, 20]:
        col = f"momentum_{n}"
        df[col] = df.groupby("symbol")["close"].transform(
            lambda s, _n=n: s.pct_change(periods=_n)
        )
        cols_added.append(col)

    # MA relationship (short MA vs long MA)
    if ma_periods and len(ma_periods) >= 2:
        short_p, long_p = sorted(ma_periods)[:2]
        short_col = f"ma_{short_p}"
        long_col = f"ma_{long_p}"
        if short_col in df.columns and long_col in df.columns:
            df["ma_cross_ratio"] = df[short_col] / df[long_col].replace(0, np.nan)
            cols_added.append("ma_cross_ratio")

    _register("momentum", cols_added)
    return df


# ================================================================== #
#  5. TIME FEATURES                                                    #
# ================================================================== #

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, minute, day-of-week, and cyclical encodings."""
    cols_added: List[str] = []

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Cyclical day-of-week encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    cols_added.extend([
        "hour", "minute", "day_of_week",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ])

    _register("time", cols_added)
    return df


# ================================================================== #
#  6. PRICE-VOLUME RELATIONSHIP FEATURES                               #
# ================================================================== #

def add_price_volume_features(
    df: pd.DataFrame,
    rolling_windows: List[int],
) -> pd.DataFrame:
    """Price-volume correlation, volume acceleration, return*volume."""
    cols_added: List[str] = []

    # Return × volume interaction
    df["return_volume"] = df["pct_return"] * df["volume"]
    cols_added.append("return_volume")

    # Volume acceleration (second derivative)
    df["volume_accel"] = df.groupby("symbol")["volume_change"].diff()
    cols_added.append("volume_accel")

    # Rolling price-volume correlation
    for w in rolling_windows:
        if w < 5:
            continue  # need enough points for meaningful correlation
        col = f"pv_corr_{w}"
        df[col] = df.groupby("symbol").apply(
            lambda g, _w=w: g["close"].rolling(_w, min_periods=_w).corr(g["volume"]),
            include_groups=False,
        ).droplevel(0).sort_index()
        cols_added.append(col)

    _register("price_volume", cols_added)
    return df


# ================================================================== #
#  MASTER FEATURE PIPELINE                                             #
# ================================================================== #

def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Run the full feature-engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed + labeled OHLCV data.
    cfg : dict
        Parsed config.yaml.

    Returns
    -------
    pd.DataFrame
        Dataframe with all engineered features appended.
    """
    feat_cfg = cfg.get("features", {})
    rolling_windows = feat_cfg.get("rolling_windows", [5, 10, 20, 50])
    rsi_period = feat_cfg.get("rsi_period", 14)
    atr_period = feat_cfg.get("atr_period", 14)
    zscore_window = feat_cfg.get("volume_zscore_window", 20)
    ma_periods = feat_cfg.get("ma_periods", [7, 14, 30])

    FEATURE_REGISTRY.clear()

    logger.info("Engineering features (rolling windows=%s) …", rolling_windows)

    df = add_price_features(df, rolling_windows, ma_periods)
    df = add_volume_features(df, rolling_windows, zscore_window)
    df = add_volatility_features(df, rolling_windows, atr_period)
    df = add_momentum_features(df, rsi_period=rsi_period, ma_periods=ma_periods)
    df = add_time_features(df)
    df = add_price_volume_features(df, rolling_windows)

    total_features = sum(len(v) for v in FEATURE_REGISTRY.values())
    logger.info(
        "Feature engineering complete: %d feature columns across %d groups.",
        total_features,
        len(FEATURE_REGISTRY),
    )
    for group, cols in FEATURE_REGISTRY.items():
        logger.info("  %-14s : %d features", group, len(cols))

    return df


def get_feature_columns(groups: Optional[List[str]] = None) -> List[str]:
    """
    Return feature column names, optionally filtered to specified groups.

    Parameters
    ----------
    groups : list of str, optional
        Feature group names (e.g. ["price", "volume"]).
        If None, return all feature columns.

    Returns
    -------
    list of str
    """
    if groups is None:
        groups = list(FEATURE_REGISTRY.keys())

    cols = []
    for g in groups:
        if g in FEATURE_REGISTRY:
            cols.extend(FEATURE_REGISTRY[g])
        elif g == "context":
            # Context / channel features — placeholder for future data
            logger.info("Feature group 'context' requested but not available in current dataset.")
        else:
            logger.warning("Unknown feature group requested: '%s'", g)
    return cols


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
    from src.labeling import create_binary_labels

    cfg = load_config()
    try:
        ohlcv, labels = load_all_data(cfg)
        ohlcv = preprocess(ohlcv)
        horizon = cfg.get("prediction_horizon_minutes", 30)
        labeled = create_binary_labels(ohlcv, labels, horizon)
        featured = engineer_features(labeled, cfg)

        print("\n--- Feature Registry ---")
        for g, cols in FEATURE_REGISTRY.items():
            print(f"  {g}: {cols}")

        print(f"\nTotal columns: {featured.shape[1]}")
        print(featured.head())
    except FileNotFoundError as exc:
        print(f"\n[WARNING] {exc}")
