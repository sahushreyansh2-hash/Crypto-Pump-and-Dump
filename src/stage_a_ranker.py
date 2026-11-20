"""
stage_a_ranker.py — Pre-Announcement Target Coin Ranking Module (Stage A).

Addresses Faculty Requirements:
1. "Compare Stage A ablations: Market-only, channel history, cross-source history and combined."
2. "CATR has zero top-1 hits and only a small HR@30 gain. Improve and validate genuine
   pre-announcement ranking."
3. "Demonstrate whether the pre-announcement ranking stage provides operational value."

Stage A computes a ranking score for every candidate coin in the trading universe
strictly BEFORE any announcement is released.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class StageARanker:
    """
    Stage A Candidate Coin Ranker.
    
    Supports 4 ablation configurations:
    - 'market_only'       : Volume accumulation z-score, illiquidity index, volatility compression
    - 'channel_history'   : Telegram group historical target frequency, group pump recency
    - 'cross_source'      : Multi-channel overlap, cross-group co-occurrence frequency
    - 'combined'          : Full multi-modal feature set (Market + Channel + Cross-Source)
    """

    def __init__(self, mode: str = "combined", random_state: int = 42):
        assert mode in ["market_only", "channel_history", "cross_source", "combined"], f"Invalid mode: {mode}"
        self.mode = mode
        self.random_state = random_state
        self.model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.08, random_state=random_state
        )
        self.feature_names_: List[str] = []
        self.is_fitted = False

    def _extract_features(
        self,
        market_df: pd.DataFrame,
        pump_labels: pd.DataFrame,
        t_eval: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Extract pre-announcement ranking features strictly available at or before t_eval.
        Zero future information is used.
        """
        symbols = market_df["symbol"].unique()
        records = []
        
        # Historical labels strictly strictly prior to t_eval
        hist_pumps = pump_labels[pump_labels["event_timestamp"] < t_eval].copy()
        
        # Cross-source mentions count prior to t_eval
        cross_source_counts = hist_pumps["symbol"].value_counts().to_dict() if not hist_pumps.empty else {}
        channel_group_counts = hist_pumps.groupby(["group", "symbol"]).size().to_dict() if "group" in hist_pumps.columns and not hist_pumps.empty else {}

        for sym in symbols:
            sym_market = market_df[(market_df["symbol"] == sym) & (market_df["timestamp"] <= t_eval)]
            if len(sym_market) < 10:
                continue
                
            recent = sym_market.tail(30)
            
            # --- 1. Market Signals ---
            # Low liquidity indicator (lower volume coins are preferred targets for pumpers)
            median_vol = recent["volume"].median()
            log_vol = np.log1p(max(1e-4, median_vol))
            
            # Pre-pump stealth accumulation (subtle volume trend vs baseline)
            vol_5 = recent["volume"].tail(5).mean()
            vol_20 = recent["volume"].mean()
            stealth_vol_ratio = (vol_5 / (vol_20 + 1e-6))
            
            # Volatility compression (quiet consolidation before explosive pump)
            recent_ret = recent["close"].pct_change().dropna()
            volatility_compression = recent_ret.tail(10).std() / (recent_ret.std() + 1e-6) if len(recent_ret) > 5 else 1.0
            
            # Orderbook depth proxy / Price range tightness
            price_range_ratio = (recent["high"].tail(5).max() - recent["low"].tail(5).min()) / (recent["close"].iloc[-1] + 1e-8)

            # --- 2. Channel History Signals ---
            sym_coin = sym.split("/")[0]
            past_pump_count = cross_source_counts.get(sym, cross_source_counts.get(sym_coin, 0))
            
            # Days since last pumped (target cycle recurrence)
            sym_past_events = hist_pumps[hist_pumps["symbol"].isin([sym, sym_coin])]
            if not sym_past_events.empty:
                last_event_time = sym_past_events["event_timestamp"].max()
                days_since_last_pump = (t_eval - last_event_time).total_seconds() / 86400.0
            else:
                days_since_last_pump = 180.0  # Default long prior
                
            # --- 3. Cross-Source / Multi-Channel Signals ---
            multi_channel_diversity = sym_past_events["channel_id"].nunique() if ("channel_id" in sym_past_events.columns and not sym_past_events.empty) else 0
            group_diversity = sym_past_events["group"].nunique() if ("group" in sym_past_events.columns and not sym_past_events.empty) else 0
            co_occurrence_score = np.log1p(past_pump_count * (group_diversity + 1))

            records.append({
                "symbol": sym,
                # Market
                "log_median_vol": float(log_vol),
                "stealth_vol_ratio": float(stealth_vol_ratio),
                "volatility_compression": float(volatility_compression),
                "price_range_ratio": float(price_range_ratio),
                # Channel History
                "past_pump_count": float(past_pump_count),
                "days_since_last_pump": float(days_since_last_pump),
                # Cross-Source
                "multi_channel_diversity": float(multi_channel_diversity),
                "group_diversity": float(group_diversity),
                "co_occurrence_score": float(co_occurrence_score),
            })
            
        feat_df = pd.DataFrame(records)
        return feat_df

    def _get_mode_columns(self) -> List[str]:
        if self.mode == "market_only":
            return ["log_median_vol", "stealth_vol_ratio", "volatility_compression", "price_range_ratio"]
        elif self.mode == "channel_history":
            return ["past_pump_count", "days_since_last_pump"]
        elif self.mode == "cross_source":
            return ["multi_channel_diversity", "group_diversity", "co_occurrence_score"]
        else:  # combined
            return [
                "log_median_vol", "stealth_vol_ratio", "volatility_compression", "price_range_ratio",
                "past_pump_count", "days_since_last_pump",
                "multi_channel_diversity", "group_diversity", "co_occurrence_score",
            ]

    def fit(self, training_events: List[Dict[str, Any]], market_df: pd.DataFrame, pump_labels: pd.DataFrame):
        """
        Fit ranker on historical events before the test period.
        """
        X_list, y_list = [], []
        cols = self._get_mode_columns()
        self.feature_names_ = cols

        for event in training_events:
            t_eval = event["announcement_time"] - pd.Timedelta(minutes=5)  # 5 min before announcement
            target_sym = event["symbol"]
            
            feats = self._extract_features(market_df, pump_labels, t_eval)
            if feats.empty:
                continue
                
            for _, row in feats.iterrows():
                is_target = 1 if (row["symbol"] == target_sym or row["symbol"].split("/")[0] == target_sym.split("/")[0]) else 0
                X_list.append(row[cols].values)
                y_list.append(is_target)

        if len(X_list) > 0 and sum(y_list) > 0:
            X = np.nan_to_num(np.array(X_list), nan=0.0)
            y = np.array(y_list)
            self.model.fit(X, y)
            self.is_fitted = True
        return self

    def rank_coins(
        self,
        market_df: pd.DataFrame,
        pump_labels: pd.DataFrame,
        t_eval: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Produce a ranked list of candidate coins at timestamp t_eval.
        Returns DataFrame with columns ['symbol', 'score', 'rank'].
        """
        feats = self._extract_features(market_df, pump_labels, t_eval)
        if feats.empty:
            return pd.DataFrame(columns=["symbol", "score", "rank"])
            
        cols = self._get_mode_columns()
        X = np.nan_to_num(feats[cols].values, nan=0.0)
        
        if self.is_fitted:
            scores = self.model.predict_proba(X)[:, 1]
        else:
            # Unfitted heuristic score (negative log vol + stealth vol ratio)
            scores = feats["stealth_vol_ratio"].values - 0.2 * feats["log_median_vol"].values
            
        feats["score"] = scores
        feats.sort_values("score", ascending=False, inplace=True)
        feats["rank"] = np.arange(1, len(feats) + 1)
        return feats[["symbol", "score", "rank"]].reset_index(drop=True)


def evaluate_stage_a_rankings(
    ranker: StageARanker,
    test_events: List[Dict[str, Any]],
    market_df: pd.DataFrame,
    pump_labels: pd.DataFrame,
    k_list: List[int] = [1, 5, 10, 20, 30],
) -> Dict[str, float]:
    """
    Evaluate Pre-Announcement Coin Ranking metrics across test pump events:
    - Hit Rate @ K (HR@K)
    - Mean Reciprocal Rank (MRR)
    - Average Rank
    """
    hits = {f"HR@{k}": 0 for k in k_list}
    reciprocal_ranks = []
    ranks = []
    
    valid_events = 0
    for event in test_events:
        t_eval = event["announcement_time"] - pd.Timedelta(minutes=5)
        target = event["symbol"]
        target_coin = target.split("/")[0]
        
        ranked_df = ranker.rank_coins(market_df, pump_labels, t_eval)
        if ranked_df.empty:
            continue
            
        valid_events += 1
        matches = ranked_df[
            ranked_df["symbol"].str.startswith(target_coin) | (ranked_df["symbol"] == target)
        ]
        
        if not matches.empty:
            best_rank = int(matches["rank"].min())
            ranks.append(best_rank)
            reciprocal_ranks.append(1.0 / best_rank)
            for k in k_list:
                if best_rank <= k:
                    hits[f"HR@{k}"] += 1
        else:
            total_universe = len(ranked_df)
            ranks.append(total_universe)
            reciprocal_ranks.append(0.0)

    n = max(1, valid_events)
    results = {k: round(v / n, 4) for k, v in hits.items()}
    results["MRR"] = round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 4)
    results["Mean_Rank"] = round(float(np.mean(ranks)) if ranks else 0.0, 2)
    results["n_test_events"] = valid_events
    return results


def run_stage_a_ablations(
    training_events: List[Dict[str, Any]],
    test_events: List[Dict[str, Any]],
    market_df: pd.DataFrame,
    pump_labels: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run complete ablation study for Stage A:
    1. Market-only
    2. Channel history
    3. Cross-source history
    4. Combined
    """
    modes = ["market_only", "channel_history", "cross_source", "combined"]
    records = []
    
    for mode in modes:
        ranker = StageARanker(mode=mode, random_state=seed)
        ranker.fit(training_events, market_df, pump_labels)
        metrics = evaluate_stage_a_rankings(ranker, test_events, market_df, pump_labels)
        metrics["Ablation_Mode"] = mode
        records.append(metrics)
        
    df = pd.DataFrame(records)
    cols = ["Ablation_Mode", "HR@1", "HR@5", "HR@10", "HR@20", "HR@30", "MRR", "Mean_Rank", "n_test_events"]
    return df[[c for c in cols if c in df.columns]]
