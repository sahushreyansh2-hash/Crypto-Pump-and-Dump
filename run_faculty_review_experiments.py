"""
run_faculty_review_experiments.py — Master Benchmark Suite for Faculty Review.

Executes all P1 and P2 Review Requirements:
1. Stage A Pre-Announcement Ranking Ablations (Market, Channel, Cross-Source, Combined)
2. Stage B Detection Model Ablations (XGBoost, Temporal MLP, Isolation Forest, Stacking)
3. Cascade vs ETASE Alone at Matched Daily False-Alarm Rates (The Key Experiment!)
4. 5-Seed Repeated Statistical Tests & 95% Confidence Intervals
5. Rolling-Origin Chronological Holdouts
6. Unseen Coins & Unseen Telegram Groups Generalization
7. Stacked Decision Feature Attribution
8. Real-Time Inference Latency Audit
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import load_config, load_ohlcv, load_pump_labels
from src.preprocessing import preprocess
from src.labeling import create_binary_labels
from src.features import engineer_features, get_feature_columns
from src.exchange_stream import build_representative_exchange_stream
from src.stage_a_ranker import StageARanker, run_stage_a_ablations
from src.stage_b_detector import StageBDetector, run_stage_b_ablations
from src.cascade import TwoStageCascade, evaluate_matched_far_experiment
from src.statistical_tests import (
    run_repeated_seeds_benchmark,
    run_rolling_origin_evaluation,
    evaluate_unseen_generalization,
)
from src.explainability import compute_stacked_decision_attribution, benchmark_inference_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FacultyReview")

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("=" * 70)
    logger.info("  FACULTY REVIEW EXPERIMENTAL BENCHMARK SUITE")
    logger.info("=" * 70)

    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))

    # 1. Load Data
    logger.info("[1/8] Loading Market Data & Verified Pump Labels...")
    raw_ohlcv = load_ohlcv(cfg["data"]["ohlcv_path"])
    clean_ohlcv = preprocess(raw_ohlcv)
    pump_labels = load_pump_labels(cfg["data"]["pump_labels_path"])
    
    labeled_df = create_binary_labels(clean_ohlcv, pump_labels, prediction_horizon_minutes=30)
    featured_df = engineer_features(labeled_df, cfg)
    
    feature_cols = get_feature_columns()
    featured_df = featured_df.dropna(subset=feature_cols).reset_index(drop=True)

    # 2. Extract Events for Stage A & Cascade
    logger.info("[2/8] Setting Up Chronological Event Windows...")
    t_min, t_max = featured_df["timestamp"].min(), featured_df["timestamp"].max()
    t_split = t_min + pd.Timedelta(days=1) if (t_max - t_min).days >= 2 else t_min + (t_max - t_min) * 0.7

    train_df = featured_df[featured_df["timestamp"] <= t_split].copy()
    test_df = featured_df[featured_df["timestamp"] > t_split].copy()

    X_train = np.clip(np.nan_to_num(train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32), nan=0.0), -1e5, 1e5)
    y_train = train_df["target"].values
    X_test = np.clip(np.nan_to_num(test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32), nan=0.0), -1e5, 1e5)
    y_test = test_df["target"].values

    # Sample pump events for Stage A validation
    test_symbols = featured_df["symbol"].unique()
    sample_events = []
    for s in test_symbols[:10]:
        s_pumps = pump_labels[pump_labels["symbol"].str.startswith(s.split("/")[0])]
        if not s_pumps.empty:
            sample_events.append({
                "symbol": s,
                "announcement_time": s_pumps["event_timestamp"].iloc[0],
            })
        else:
            sample_events.append({
                "symbol": s,
                "announcement_time": t_max - pd.Timedelta(hours=4),
            })

    train_events = sample_events[:len(sample_events)//2]
    test_events = sample_events[len(sample_events)//2:]

    # 3. Stage A Ablation Study
    logger.info("[3/8] Running Stage A Ranking Ablations (Market, Channel, Cross-Source, Combined)...")
    stage_a_ablation_df = run_stage_a_ablations(train_events, test_events, featured_df, pump_labels, seed=42)
    stage_a_out = RESULTS_DIR / "stage_a_ablation_results.csv"
    stage_a_ablation_df.to_csv(stage_a_out, index=False)
    logger.info("Stage A Ablations saved to %s", stage_a_out)

    # 4. Stage B Ablation Study
    logger.info("[4/8] Running Stage B Model Ablations (XGBoost, MLP, Isolation Forest, Stacking)...")
    stage_b_ablation_df = run_stage_b_ablations(X_train, y_train, X_test, y_test, threshold=0.5, seed=42)
    stage_b_out = RESULTS_DIR / "stage_b_ablation_results.csv"
    stage_b_ablation_df.to_csv(stage_b_out, index=False)
    logger.info("Stage B Ablations saved to %s", stage_b_out)

    # 5. Build Representative Exchange-Wide Stream
    logger.info("[5/8] Generating Exchange-Wide Trading Stream for Direct False-Alarm Audit...")
    pumped_syms = list(featured_df["symbol"].unique())
    exchange_stream_df, stream_meta = build_representative_exchange_stream(
        pumped_symbols=pumped_syms,
        n_non_pumped=30,
        n_chunks=720,  # 12 hours of exchange trading
        chunk_seconds=60,
        seed=42,
    )
    # Engineer features on exchange stream
    exchange_stream_featured = engineer_features(exchange_stream_df, cfg)
    exchange_stream_featured = exchange_stream_featured.dropna(subset=feature_cols).reset_index(drop=True)

    # 6. KEY EXPERIMENT: Matched False-Alarm Rate Comparison (ETASE Alone vs Cascade)
    logger.info("[6/8] Running KEY EXPERIMENT: ETASE Alone vs Two-Stage Cascade at Matched FAR...")
    etase_detector = StageBDetector(model_type="xgboost", random_state=42)
    etase_detector.fit(X_train, y_train)

    cascade_k10 = TwoStageCascade(top_k=10, stage_a_mode="combined", stage_b_model="xgboost", random_state=42)
    cascade_k10.fit(train_events, featured_df, pump_labels, X_train, y_train)

    cascade_k20 = TwoStageCascade(top_k=20, stage_a_mode="combined", stage_b_model="xgboost", random_state=42)
    cascade_k20.fit(train_events, featured_df, pump_labels, X_train, y_train)

    matched_far_df = evaluate_matched_far_experiment(
        stream_df=exchange_stream_featured,
        test_events=test_events,
        feature_cols=feature_cols,
        etase_detector=etase_detector,
        cascade_k10=cascade_k10,
        cascade_k20=cascade_k20,
        target_far_budgets=[5.0, 10.0, 20.0, 30.0, 40.0],
    )
    matched_far_out = RESULTS_DIR / "cascade_matched_far_comparison.csv"
    matched_far_df.to_csv(matched_far_out, index=False)
    logger.info("Matched FAR Comparison saved to %s", matched_far_out)

    # 7. Repeated Seeds & Statistical Tests
    logger.info("[7/8] Conducting 5-Seed Repeated Statistical Tests & Confidence Intervals...")
    seeds_df, sig_report = run_repeated_seeds_benchmark(X_train, y_train, X_test, y_test, seeds=[42, 43, 44, 45, 46])
    seeds_out = RESULTS_DIR / "repeated_seeds_confidence_intervals.csv"
    seeds_df.to_csv(seeds_out, index=False)

    rolling_origin_df = run_rolling_origin_evaluation(featured_df, feature_cols, n_splits=3)
    rolling_out = RESULTS_DIR / "rolling_origin_results.csv"
    rolling_origin_df.to_csv(rolling_out, index=False)

    unseen_df = evaluate_unseen_generalization(featured_df, pump_labels, feature_cols)
    unseen_out = RESULTS_DIR / "unseen_generalization_results.csv"
    unseen_df.to_csv(unseen_out, index=False)

    # 8. Explainability & Real-Time Inference Latency Audit
    logger.info("[8/8] Computing Full Stacked Decision Attribution & Latency Audit...")
    attr_df = compute_stacked_decision_attribution(etase_detector.model, X_test[:100], feature_cols, n_top=12)
    attr_out = RESULTS_DIR / "stacked_decision_attribution.csv"
    attr_df.to_csv(attr_out, index=False)

    latency_stats = benchmark_inference_latency(etase_detector.model, n_features=len(feature_cols))
    latency_out = RESULTS_DIR / "inference_latency_benchmark.json"
    with open(latency_out, "w", encoding="utf-8") as f:
        json.dump(latency_stats, f, indent=2)

    logger.info("=" * 70)
    logger.info("  ALL FACULTY REVIEW EXPERIMENTS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
