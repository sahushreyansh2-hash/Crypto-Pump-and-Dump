"""
ablation.py — Ablation study comparing feature-group subsets.

Experiments
-----------
A. Market signals only      (price, volume, volatility, price_volume)
B. Temporal signals only    (time, momentum)
C. Market + Temporal        (all of A + B)
D. Market + Temporal + Context/History
   → If context data is unavailable, this experiment is marked as such
     rather than generating fabricated results.

For each experiment, a separate XGBoost (or Random Forest fallback) model
is trained and evaluated on the same chronological test set.

Results are saved to ``results/ablation_results.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation import compute_metrics, find_best_threshold, compute_lead_time
from src.features import FEATURE_REGISTRY, get_feature_columns

logger = logging.getLogger(__name__)


def _get_model_class(cfg: dict):
    """Return the model class and params for ablation (XGBoost or RF fallback)."""
    try:
        import xgboost as xgb

        xgb_cfg = cfg.get("xgboost", {})
        params = {
            "n_estimators": xgb_cfg.get("n_estimators", 300),
            "max_depth": xgb_cfg.get("max_depth", 8),
            "learning_rate": xgb_cfg.get("learning_rate", 0.05),
            "eval_metric": xgb_cfg.get("eval_metric", "aucpr"),
            "use_label_encoder": False,
            "random_state": cfg.get("random_seed", 42),
            "verbosity": 0,
        }
        return xgb.XGBClassifier, params, "XGBoost"
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier

        rf_cfg = cfg.get("random_forest", {})
        params = {
            "n_estimators": rf_cfg.get("n_estimators", 300),
            "max_depth": rf_cfg.get("max_depth", 12),
            "class_weight": "balanced",
            "min_samples_leaf": rf_cfg.get("min_samples_leaf", 20),
            "random_state": cfg.get("random_seed", 42),
            "n_jobs": -1,
        }
        return RandomForestClassifier, params, "RandomForest"


def run_ablation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    time_to_pump_test: Optional[np.ndarray],
    cfg: dict,
) -> pd.DataFrame:
    """
    Run ablation study over feature-group combinations.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices (all columns present; filtering done here).
    y_train, y_val, y_test : array-like
        Binary labels.
    time_to_pump_test : array-like or None
        Minutes to next pump for lead-time analysis.
    cfg : dict
        Parsed config.yaml.

    Returns
    -------
    pd.DataFrame
        Ablation results with one row per experiment.
    """
    experiments = cfg.get("ablation", {}).get("experiments", [])
    if not experiments:
        logger.warning("No ablation experiments defined in config.")
        return pd.DataFrame()

    ModelClass, model_params, model_name = _get_model_class(cfg)
    logger.info("Ablation study using %s", model_name)

    out_dir = Path(cfg["data"].get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold_steps = cfg.get("evaluation", {}).get("threshold_search_steps", 50)
    all_results: List[Dict[str, Any]] = []

    for exp in experiments:
        exp_name = exp["name"]
        feature_groups = exp["feature_groups"]

        logger.info("--- Ablation: %s  groups=%s ---", exp_name, feature_groups)

        # Check if any requested group is unavailable
        if "context" in feature_groups and "context" not in FEATURE_REGISTRY:
            logger.info(
                "  Experiment '%s' requires 'context' features which are "
                "NOT available in the current dataset. "
                "Marking as unavailable (not fabricating results).",
                exp_name,
            )
            all_results.append({
                "experiment": exp_name,
                "feature_groups": str(feature_groups),
                "n_features": 0,
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "pr_auc": float("nan"),
                "roc_auc": float("nan"),
                "false_alarm_rate": float("nan"),
                "detection_rate": float("nan"),
                "mean_lead_time_min": float("nan"),
                "available": False,
                "note": "Context/channel features not available in current dataset.",
            })
            continue

        # Get feature columns for this experiment
        cols = get_feature_columns(feature_groups)
        # Filter to columns actually present in X_train
        cols = [c for c in cols if c in X_train.columns]

        if len(cols) == 0:
            logger.warning("  No valid features for experiment '%s'.", exp_name)
            continue

        logger.info("  Using %d features: %s", len(cols), cols[:5])

        # Prepare data
        Xtr = np.nan_to_num(X_train[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values)
        Xv = np.nan_to_num(X_val[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values)
        Xt = np.nan_to_num(X_test[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values)

        # Handle class imbalance for XGBoost
        params = model_params.copy()
        if "XGB" in model_name:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            if n_pos > 0:
                params["scale_pos_weight"] = n_neg / n_pos

        # Train
        model = ModelClass(**params)
        model.fit(Xtr, y_train)

        # Predict probabilities
        val_probs = model.predict_proba(Xv)[:, 1]
        test_probs = model.predict_proba(Xt)[:, 1]

        # Find best threshold on validation
        best_t, _ = find_best_threshold(y_val, val_probs, threshold_steps)

        # Evaluate on test
        metrics = compute_metrics(y_test, test_probs, threshold=best_t)

        # Lead time
        lt_mean = float("nan")
        if time_to_pump_test is not None:
            lt = compute_lead_time(y_test, test_probs, time_to_pump_test, threshold=best_t)
            lt_mean = lt["mean_lead_time"]

        result = {
            "experiment": exp_name,
            "feature_groups": str(feature_groups),
            "n_features": len(cols),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "false_alarm_rate": metrics["false_alarm_rate"],
            "detection_rate": metrics["detection_rate"],
            "mean_lead_time_min": lt_mean,
            "threshold": best_t,
            "available": True,
            "note": "",
        }
        all_results.append(result)

        logger.info(
            "  %s — F1=%.4f  PR-AUC=%.4f  ROC-AUC=%.4f",
            exp_name,
            metrics["f1"],
            metrics["pr_auc"],
            metrics["roc_auc"],
        )

    # Save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "ablation_results.csv", index=False)
    logger.info("Ablation results saved to %s", out_dir / "ablation_results.csv")

    return results_df


# ------------------------------------------------------------------ #
#  CLI quick-test                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info(
        "Ablation module loaded. Run via main.py or supply train/val/test data."
    )
