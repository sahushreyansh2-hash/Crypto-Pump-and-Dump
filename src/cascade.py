"""
cascade.py — Two-Stage Cascade Pipeline & Matched False-Alarm Evaluation.

Addresses Faculty Mandates:
1. "Most important experiment: Compare ETASE alone against the complete two-stage cascade
   at the same daily false-alarm budget, using strictly future-held-out events."
2. "Determine whether the pre-announcement ranking stage actually provides operational value."
3. "Measure false alarms directly on representative exchange-wide trading streams."
4. "Explain precisely when each chunk becomes available (chunk close) and calculate detection
   delay strictly from data availability time."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from src.stage_a_ranker import StageARanker
from src.stage_b_detector import StageBDetector

logger = logging.getLogger(__name__)


@dataclass
class AlertRecord:
    symbol: str
    chunk_start: pd.Timestamp
    chunk_end: pd.Timestamp           # Strict data availability timestamp
    alert_time: pd.Timestamp          # Timestamped at chunk_end
    announcement_time: pd.Timestamp
    probability: float
    is_true_positive: bool
    detection_delay_seconds: float    # alert_time - announcement_time
    lead_time_seconds: float          # announcement_time - alert_time (if pre-announcement)
    is_pre_onset: bool                # True if alert_time <= announcement_time


class TwoStageCascade:
    """
    Two-Stage Cascade Architecture:
    - Stage A (Pre-Announcement): Ranks all trading assets and selects Top-K candidates.
    - Stage B (Post-Onset / Early-Spike): Evaluates candidate stream for anomalous trading.
    """

    def __init__(
        self,
        top_k: int = 10,
        stage_a_mode: str = "combined",
        stage_b_model: str = "xgboost",
        random_state: int = 42,
    ):
        self.top_k = top_k
        self.stage_a = StageARanker(mode=stage_a_mode, random_state=random_state)
        self.stage_b = StageBDetector(model_type=stage_b_model, random_state=random_state)
        self.random_state = random_state

    def fit(
        self,
        training_events: List[Dict[str, Any]],
        market_df: pd.DataFrame,
        pump_labels: pd.DataFrame,
        X_train_b: np.ndarray,
        y_train_b: np.ndarray,
    ):
        """Fit Stage A and Stage B on chronological training data."""
        self.stage_a.fit(training_events, market_df, pump_labels)
        self.stage_b.fit(X_train_b, y_train_b)
        return self

    def predict_stream(
        self,
        stream_df: pd.DataFrame,
        pump_labels: pd.DataFrame,
        feature_cols: List[str],
        threshold: float = 0.5,
        t_rank_eval: Optional[pd.Timestamp] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Process exchange trading stream through the cascade.
        1. Stage A determines Top-K candidate symbols.
        2. Stage B executes inference on candidate symbols at each chunk.
        """
        if t_rank_eval is None:
            t_rank_eval = stream_df["timestamp"].min()

        # Stage A: Rank coins and select Top-K
        ranked_df = self.stage_a.rank_coins(stream_df, pump_labels, t_rank_eval)
        top_k_candidates = ranked_df.head(self.top_k)["symbol"].tolist() if not ranked_df.empty else []

        # Filter stream to candidate symbols
        candidate_stream = stream_df[stream_df["symbol"].isin(top_k_candidates)].copy()
        
        if candidate_stream.empty or not feature_cols:
            stream_df["cascade_prob"] = 0.0
            stream_df["cascade_alert"] = 0
            return stream_df, top_k_candidates

        X = np.clip(np.nan_to_num(candidate_stream[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32), nan=0.0), -1e5, 1e5)
        probs = self.stage_b.predict_proba(X)
        candidate_stream["cascade_prob"] = probs
        candidate_stream["cascade_alert"] = (probs >= threshold).astype(int)

        # Merge back to full stream
        result_df = pd.merge(
            stream_df,
            candidate_stream[["timestamp", "symbol", "cascade_prob", "cascade_alert"]],
            on=["timestamp", "symbol"],
            how="left",
        )
        result_df["cascade_prob"] = result_df["cascade_prob"].fillna(0.0)
        result_df["cascade_alert"] = result_df["cascade_alert"].fillna(0).astype(int)

        return result_df, top_k_candidates


def find_threshold_for_far_budget(
    probs: np.ndarray,
    negative_mask: np.ndarray,
    target_daily_far: float,
    stream_duration_days: float = 1.0,
) -> float:
    """
    Calibrate classification threshold to achieve an exact target daily false alarm count.
    """
    neg_probs = np.sort(probs[negative_mask])
    n_neg = len(neg_probs)
    if n_neg == 0:
        return 0.5

    # Max allowed false alarms over stream duration
    max_fp = target_daily_far * stream_duration_days
    if max_fp >= n_neg:
        return float(neg_probs[0])
    
    # Pick threshold index corresponding to target FP count
    idx = int(n_neg - max_fp)
    idx = max(0, min(idx, n_neg - 1))
    threshold = float(neg_probs[idx])
    return max(0.01, min(0.99, threshold))


def evaluate_matched_far_experiment(
    stream_df: pd.DataFrame,
    test_events: List[Dict[str, Any]],
    feature_cols: List[str],
    etase_detector: Any,
    cascade_k10: TwoStageCascade,
    cascade_k20: TwoStageCascade,
    target_far_budgets: List[float] = [5.0, 10.0, 20.0, 30.0, 40.0],
    chunk_seconds: int = 60,
) -> pd.DataFrame:
    """
    FACULTY REVIEW CORE EXPERIMENT:
    Compare ETASE alone against complete Two-Stage Cascade at MATCHED daily false-alarm rates.
    
    Evaluates:
    - Event Recall (% of target pump events flagged before or within onset window)
    - Pre-Onset Alerts (Alerts fired strictly at chunk close before announcement)
    - Mean Detection Delay (seconds from announcement, measured strictly at data availability)
    - Daily False Alarm Count (measured directly across entire exchange universe)
    """
    records = []
    stream_duration_days = (stream_df["timestamp"].max() - stream_df["timestamp"].min()).total_seconds() / 86400.0
    stream_duration_days = max(1.0, stream_duration_days)

    # Base features
    X_full = np.clip(np.nan_to_num(stream_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32), nan=0.0), -1e5, 1e5)
    
    # ETASE baseline probabilities
    etase_probs = etase_detector.predict_proba(X_full)
    
    # Cascade K=10 & K=20 predictions
    pump_labels = pd.DataFrame(test_events)
    if "event_timestamp" not in pump_labels.columns and "announcement_time" in pump_labels.columns:
        pump_labels["event_timestamp"] = pump_labels["announcement_time"]
    t0 = stream_df["timestamp"].min()
    
    k10_df, k10_candidates = cascade_k10.predict_stream(stream_df, pump_labels, feature_cols, threshold=0.5, t_rank_eval=t0)
    k10_probs = k10_df["cascade_prob"].values

    k20_df, k20_candidates = cascade_k20.predict_stream(stream_df, pump_labels, feature_cols, threshold=0.5, t_rank_eval=t0)
    k20_probs = k20_df["cascade_prob"].values

    # Negative mask for false alarms (background non-pump chunks)
    neg_mask = (stream_df["is_pump_window"] == False).values

    models = [
        ("ETASE Alone", etase_probs),
        ("Cascade (K=10)", k10_probs),
        ("Cascade (K=20)", k20_probs),
    ]

    for target_far in target_far_budgets:
        for model_name, probs in models:
            # Calibrate threshold at matched FAR budget
            thresh = find_threshold_for_far_budget(probs, neg_mask, target_far, stream_duration_days)
            alerts = probs >= thresh

            # Measured actual false alarms per day
            actual_fp = (alerts & neg_mask).sum()
            actual_far_daily = actual_fp / stream_duration_days

            # Evaluate on test pump events
            detected_events = 0
            pre_onset_alerts = 0
            delays = []

            for event in test_events:
                sym = event["symbol"]
                t_ann = event["announcement_time"]
                
                # Check 30-minute window around announcement: [t_ann - 15m, t_ann + 15m]
                window = stream_df[
                    (stream_df["symbol"] == sym) &
                    (stream_df["timestamp"] >= t_ann - pd.Timedelta(minutes=15)) &
                    (stream_df["timestamp"] <= t_ann + pd.Timedelta(minutes=15))
                ]
                
                if window.empty:
                    continue
                    
                w_indices = window.index
                w_alerts = alerts[w_indices]
                
                if w_alerts.any():
                    detected_events += 1
                    # First alert chunk
                    first_alert_idx = window[w_alerts].index[0]
                    first_alert_time = stream_df.loc[first_alert_idx, "chunk_end"]  # Strict availability!
                    
                    delay = (first_alert_time - t_ann).total_seconds()
                    delays.append(delay)
                    
                    if first_alert_time <= t_ann:
                        pre_onset_alerts += 1

            n_events = max(1, len(test_events))
            recall = detected_events / n_events
            mean_delay = float(np.mean(delays)) if delays else np.nan

            records.append({
                "Configuration": model_name,
                "Target_FAR_Budget": target_far,
                "Actual_FAR_Daily": round(actual_far_daily, 1),
                "Event_Recall": f"{recall * 100:.1f}%",
                "Pre_Onset_Alerts": pre_onset_alerts,
                "Mean_Detection_Delay_s": round(mean_delay, 1) if not np.isnan(mean_delay) else "N/A",
                "Operating_Threshold": round(thresh, 4),
            })

    return pd.DataFrame(records)
