"""
main.py — Orchestrator for the Crypto Pump-and-Dump Early Detection pipeline.

Usage
-----
    python main.py                      # run full pipeline (default config)
    python main.py --config myconf.yaml # custom config
    python main.py --phase 1            # run only a specific phase

Phases
------
    1  Data loading & validation
    2  Preprocessing & labeling
    3  Feature engineering
    4  Chronological split
    5  Rule-based + Isolation Forest + classical ML baselines
    6  LSTM / GRU temporal model
    7  ETASE framework
    8  Evaluation & lead-time analysis
    9  Ablation study
   10  Visualization
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config, load_all_data
from src.preprocessing import preprocess
from src.labeling import create_binary_labels, label_statistics
from src.features import engineer_features, get_feature_columns, FEATURE_REGISTRY
from src.split import chronological_split, split_summary, save_split_info
from src.evaluation import evaluate_all_models
from src.ablation import run_ablation

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def setup_logging() -> None:
    """Configure console + file logging."""
    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    log_fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROJECT_ROOT / "results" / "pipeline.log", mode="a"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crypto Pump-and-Dump Early Detection Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Run only a specific phase (1-10). Omit to run all.",
    )
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  State container for passing data between phases                     #
# ------------------------------------------------------------------ #

class PipelineState:
    """Mutable container to pass data between phases."""
    def __init__(self):
        self.ohlcv = None
        self.pump_labels = None
        self.labeled = None
        self.featured = None
        self.train = None
        self.val = None
        self.test = None
        self.model_results = {}  # model_name -> {model, val_probs, test_probs, ...}
        self.summary_df = None


# ------------------------------------------------------------------ #
#  Phase implementations                                               #
# ------------------------------------------------------------------ #

def phase_1(cfg: dict, state: PipelineState) -> None:
    """Phase 1: Load & validate raw data."""
    log = logging.getLogger("phase1")
    log.info("=" * 60)
    log.info("PHASE 1 — Data Loading & Validation")
    log.info("=" * 60)

    state.ohlcv, state.pump_labels = load_all_data(cfg)
    log.info("OHLCV  : %d rows, %d symbols", len(state.ohlcv), state.ohlcv["symbol"].nunique())
    log.info("Labels : %d pump events", len(state.pump_labels))
    log.info("Phase 1 complete.")


def phase_2(cfg: dict, state: PipelineState) -> None:
    """Phase 2: Preprocessing & labeling."""
    log = logging.getLogger("phase2")
    log.info("=" * 60)
    log.info("PHASE 2 — Preprocessing & Labeling")
    log.info("=" * 60)

    state.ohlcv = preprocess(state.ohlcv)

    horizon = cfg.get("prediction_horizon_minutes", 30)
    state.labeled = create_binary_labels(state.ohlcv, state.pump_labels, horizon)

    stats = label_statistics(state.labeled)
    log.info("Label distribution:\n%s", stats.to_string(index=False))
    stats.to_csv(Path(cfg["data"]["results_dir"]) / "label_statistics.csv", index=False)
    log.info("Phase 2 complete.")


def phase_3(cfg: dict, state: PipelineState) -> None:
    """Phase 3: Feature engineering."""
    log = logging.getLogger("phase3")
    log.info("=" * 60)
    log.info("PHASE 3 — Feature Engineering")
    log.info("=" * 60)

    state.featured = engineer_features(state.labeled, cfg)

    # Drop rows with NaN from rolling windows
    feature_cols = get_feature_columns()
    n_before = len(state.featured)
    state.featured.dropna(subset=feature_cols, inplace=True)
    state.featured.reset_index(drop=True, inplace=True)
    log.info("Dropped %d rows with NaN features (from rolling windows).", n_before - len(state.featured))

    # Save processed data
    proc_dir = Path(cfg["data"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    state.featured.to_csv(proc_dir / "featured_data.csv", index=False)
    log.info("Featured data saved (%d rows, %d cols).", *state.featured.shape)
    log.info("Phase 3 complete.")


def phase_4(cfg: dict, state: PipelineState) -> None:
    """Phase 4: Chronological split."""
    log = logging.getLogger("phase4")
    log.info("=" * 60)
    log.info("PHASE 4 — Chronological Split")
    log.info("=" * 60)

    state.train, state.val, state.test = chronological_split(
        state.featured,
        cfg.get("train_ratio", 0.70),
        cfg.get("validation_ratio", 0.15),
        cfg.get("test_ratio", 0.15),
    )

    summary = split_summary(state.train, state.val, state.test)
    log.info("Split summary:\n%s", summary.to_string(index=False))
    save_split_info(summary, cfg["data"].get("results_dir", "results"))
    log.info("Phase 4 complete.")


def phase_5(cfg: dict, state: PipelineState) -> None:
    """Phase 5: Rule-based + Isolation Forest + Classical ML baselines."""
    log = logging.getLogger("phase5")
    log.info("=" * 60)
    log.info("PHASE 5 — Baseline Models")
    log.info("=" * 60)

    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in state.train.columns]

    X_train = state.train[feature_cols].fillna(0)
    y_train = state.train["target"].values
    X_val = state.val[feature_cols].fillna(0)
    y_val = state.val["target"].values
    X_test = state.test[feature_cols].fillna(0)
    y_test = state.test["target"].values

    # --- Baselines ---
    from src.baselines import run_all_baselines
    baseline_results = run_all_baselines(X_train, y_train, X_val, X_test, cfg)

    for name, res in baseline_results.items():
        # Store val targets for threshold tuning
        res["val_targets"] = y_val
        res["test_targets"] = y_test
        state.model_results[name] = res

    # --- Isolation Forest ---
    from src.anomaly_detection import train_isolation_forest
    iso_model, iso_val_probs, iso_test_probs = train_isolation_forest(
        X_train, X_val, X_test, cfg
    )
    state.model_results["isolation_forest"] = {
        "model": iso_model,
        "val_probs": iso_val_probs,
        "val_targets": y_val,
        "test_probs": iso_test_probs,
        "test_targets": y_test,
    }

    log.info("Phase 5 complete — %d models trained.", len(state.model_results))


def phase_6(cfg: dict, state: PipelineState) -> None:
    """Phase 6: LSTM / GRU temporal model."""
    log = logging.getLogger("phase6")
    log.info("=" * 60)
    log.info("PHASE 6 — Temporal Model (LSTM/GRU)")
    log.info("=" * 60)

    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in state.train.columns]

    X_train = state.train[feature_cols].fillna(0).values
    y_train = state.train["target"].values
    X_val = state.val[feature_cols].fillna(0).values
    y_val = state.val["target"].values
    X_test = state.test[feature_cols].fillna(0).values
    y_test = state.test["target"].values

    try:
        from src.temporal_model import run_temporal_model
        result = run_temporal_model(X_train, y_train, X_val, y_val, X_test, y_test, cfg)
        state.model_results["temporal_lstm"] = result
        log.info("Temporal model training complete.")
    except Exception as e:
        log.error("Temporal model failed: %s", e, exc_info=True)


def phase_7(cfg: dict, state: PipelineState) -> None:
    """Phase 7: ETASE framework."""
    log = logging.getLogger("phase7")
    log.info("=" * 60)
    log.info("PHASE 7 — ETASE Framework")
    log.info("=" * 60)

    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in state.train.columns]

    X_train = state.train[feature_cols].fillna(0)
    y_train = state.train["target"].values
    X_val = state.val[feature_cols].fillna(0)
    y_val = state.val["target"].values
    X_test = state.test[feature_cols].fillna(0)
    y_test = state.test["target"].values

    try:
        from src.etase import run_etase
        result = run_etase(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, cfg)
        state.model_results["etase"] = result
        log.info("ETASE training complete.")
    except Exception as e:
        log.error("ETASE failed: %s", e, exc_info=True)


def phase_8(cfg: dict, state: PipelineState) -> None:
    """Phase 8: Evaluation & lead-time analysis."""
    log = logging.getLogger("phase8")
    log.info("=" * 60)
    log.info("PHASE 8 — Evaluation")
    log.info("=" * 60)

    y_test = state.test["target"].values
    ttp = state.test.get("time_to_next_pump_min")
    ttp_arr = ttp.values if ttp is not None else None

    threshold_steps = cfg.get("evaluation", {}).get("threshold_search_steps", 50)

    state.summary_df = evaluate_all_models(
        state.model_results,
        y_test,
        time_to_pump=ttp_arr,
        threshold_search_steps=threshold_steps,
        output_dir=cfg["data"].get("results_dir", "results"),
    )

    log.info("Evaluation results:\n%s", state.summary_df.to_string(index=False))
    log.info("Phase 8 complete.")


def phase_9(cfg: dict, state: PipelineState) -> None:
    """Phase 9: Ablation study."""
    log = logging.getLogger("phase9")
    log.info("=" * 60)
    log.info("PHASE 9 — Ablation Study")
    log.info("=" * 60)

    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in state.train.columns]

    X_train = state.train[feature_cols].fillna(0)
    y_train = state.train["target"].values
    X_val = state.val[feature_cols].fillna(0)
    y_val = state.val["target"].values
    X_test = state.test[feature_cols].fillna(0)
    y_test = state.test["target"].values

    ttp = state.test.get("time_to_next_pump_min")
    ttp_arr = ttp.values if ttp is not None else None

    # Need FEATURE_REGISTRY populated — should be from phase 3
    ablation_df = run_ablation(
        pd.DataFrame(X_train, columns=feature_cols),
        y_train,
        pd.DataFrame(X_val, columns=feature_cols),
        y_val,
        pd.DataFrame(X_test, columns=feature_cols),
        y_test,
        ttp_arr,
        cfg,
    )

    log.info("Ablation results:\n%s", ablation_df.to_string(index=False))
    log.info("Phase 9 complete.")


def phase_10(cfg: dict, state: PipelineState) -> None:
    """Phase 10: Visualization."""
    log = logging.getLogger("phase10")
    log.info("=" * 60)
    log.info("PHASE 10 — Visualization")
    log.info("=" * 60)

    from src.visualization import generate_all_plots

    if state.summary_df is None:
        log.warning("No summary_df — running evaluation first.")
        phase_8(cfg, state)

    generate_all_plots(
        df=state.featured,
        pump_events=state.pump_labels,
        model_results=state.model_results,
        summary_df=state.summary_df,
        test_df=state.test,
        cfg=cfg,
    )
    log.info("Phase 10 complete.")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

PHASES = {
    1: phase_1,
    2: phase_2,
    3: phase_3,
    4: phase_4,
    5: phase_5,
    6: phase_6,
    7: phase_7,
    8: phase_8,
    9: phase_9,
    10: phase_10,
}


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("Crypto Pump-and-Dump Early Detection Pipeline")
    logger.info("=" * 60)
    logger.info("Config : %s", args.config)
    logger.info("Phase  : %s", args.phase or "ALL")

    cfg = load_config(args.config)
    state = PipelineState()

    if args.phase is not None:
        # Run a single phase — caller is responsible for prerequisite data
        if args.phase not in PHASES:
            logger.error("Invalid phase: %d. Must be 1-10.", args.phase)
            sys.exit(1)
        # For single-phase runs, attempt to load processed data if needed
        if args.phase >= 2 and state.ohlcv is None:
            phase_1(cfg, state)
        if args.phase >= 3 and state.labeled is None:
            phase_2(cfg, state)
        if args.phase >= 4 and state.featured is None:
            phase_3(cfg, state)
        if args.phase >= 5 and state.train is None:
            phase_4(cfg, state)

        PHASES[args.phase](cfg, state)
    else:
        # Run all phases sequentially
        for phase_num in sorted(PHASES.keys()):
            try:
                PHASES[phase_num](cfg, state)
            except Exception as e:
                logger.error(
                    "Phase %d failed: %s", phase_num, e, exc_info=True
                )
                logger.info("Continuing to next phase …")

    logger.info("=" * 60)
    logger.info("Pipeline finished.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
