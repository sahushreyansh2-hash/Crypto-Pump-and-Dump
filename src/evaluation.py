"""
evaluation.py — Unified evaluation module for all models.

Metrics computed
----------------
- Precision, Recall, F1-score
- PR-AUC (Precision-Recall Area Under Curve)
- ROC-AUC
- False Alarm Rate  (FP / (FP + TN))
- Detection Rate    (TP / (TP + FN))  = Recall
- Confusion Matrix
- Lead Time analysis (how early the model flags *before* the pump event)

Lead-Time Methodology
---------------------
For every true-positive prediction (model probability >= threshold AND
actual target == 1), we measure ``time_to_next_pump_min`` which was
computed during labeling.  This tells us how many minutes before the
pump event the warning was raised.

If the model never produces a true positive, lead time is reported as
unavailable rather than fabricated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Core metric computation                                             #
# ------------------------------------------------------------------ #

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a single model.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth binary labels.
    y_prob : array-like, shape (n,)
        Predicted probabilities for the positive class.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict
        Keys: precision, recall, f1, pr_auc, roc_auc, false_alarm_rate,
        detection_rate, accuracy, threshold, confusion_matrix.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=float), nan=0.0)

    # Handle edge cases
    if len(y_true) == 0 or y_true.sum() == 0:
        logger.warning("No positive samples in y_true — metrics may be undefined.")
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true.")

    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Standard metrics
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # PR-AUC
    if len(np.unique(y_true)) >= 2:
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = auc(pr_rec, pr_prec)
    else:
        pr_auc_val = float("nan")

    # ROC-AUC
    try:
        roc_auc_val = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc_val = float("nan")

    # False alarm rate = FP / (FP + TN)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Detection rate = TP / (TP + FN) = Recall
    det_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc_val,
        "roc_auc": roc_auc_val,
        "false_alarm_rate": far,
        "detection_rate": det_rate,
        "accuracy": acc,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion_matrix": cm,
    }


# ------------------------------------------------------------------ #
#  Threshold tuning (maximize F1 on validation set)                    #
# ------------------------------------------------------------------ #

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    steps: int = 50,
) -> Tuple[float, float]:
    """
    Search for the threshold that maximizes F1-score.

    Returns
    -------
    (best_threshold, best_f1) : tuple
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    logger.info("Best threshold: %.4f (F1=%.4f)", best_t, best_f1)
    return best_t, best_f1


# ------------------------------------------------------------------ #
#  Lead-Time Analysis                                                  #
# ------------------------------------------------------------------ #

def compute_lead_time(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    time_to_pump: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Measure how early the model raises a warning before the pump event.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Predicted probabilities.
    time_to_pump : array-like
        Minutes until the next pump event (from labeling.py).
        NaN if no upcoming pump.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict
        lead_times : list of float — minutes before pump for each TP
        mean_lead_time, median_lead_time, min_lead_time, max_lead_time
        n_true_positives : int
        available : bool — False if no TPs exist (results not fabricated)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=float), nan=0.0)
    time_to_pump = np.asarray(time_to_pump, dtype=float)

    y_pred = (y_prob >= threshold).astype(int)

    # True positives: predicted 1 AND actual 1
    tp_mask = (y_pred == 1) & (y_true == 1)
    lead_times = time_to_pump[tp_mask]

    # Remove NaNs (shouldn't happen for TPs but be safe)
    lead_times = lead_times[~np.isnan(lead_times)]

    if len(lead_times) == 0:
        logger.warning(
            "No true-positive predictions — lead-time analysis is unavailable. "
            "This is reported honestly; results are NOT fabricated."
        )
        return {
            "lead_times": [],
            "mean_lead_time": float("nan"),
            "median_lead_time": float("nan"),
            "min_lead_time": float("nan"),
            "max_lead_time": float("nan"),
            "n_true_positives": 0,
            "available": False,
        }

    return {
        "lead_times": lead_times.tolist(),
        "mean_lead_time": float(np.mean(lead_times)),
        "median_lead_time": float(np.median(lead_times)),
        "min_lead_time": float(np.min(lead_times)),
        "max_lead_time": float(np.max(lead_times)),
        "n_true_positives": int(len(lead_times)),
        "available": True,
    }


# ------------------------------------------------------------------ #
#  Multi-model evaluation                                              #
# ------------------------------------------------------------------ #

def evaluate_all_models(
    model_results: Dict[str, Dict],
    y_true: np.ndarray,
    time_to_pump: Optional[np.ndarray] = None,
    threshold_search_steps: int = 50,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Evaluate all models and save results.

    Parameters
    ----------
    model_results : dict
        Mapping of model_name -> {test_probs: np.ndarray, ...}
    y_true : array-like
        Ground-truth labels for the test set.
    time_to_pump : array-like, optional
        Minutes to next pump for lead-time analysis.
    threshold_search_steps : int
        Granularity for threshold search.
    output_dir : str
        Directory to save results.

    Returns
    -------
    pd.DataFrame
        Metrics for all models.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, Any]] = []

    for model_name, result in model_results.items():
        logger.info("Evaluating model: %s", model_name)

        test_probs = result.get("test_probs")
        if test_probs is None:
            logger.warning("  No test_probs for %s — skipping.", model_name)
            continue

        # Align lengths (temporal models may produce shorter outputs)
        y_eval = y_true
        ttp_eval = time_to_pump
        if len(test_probs) < len(y_true):
            offset = len(y_true) - len(test_probs)
            y_eval = y_true[offset:]
            if ttp_eval is not None:
                ttp_eval = time_to_pump[offset:]

        # Find best threshold on validation set if available
        val_probs = result.get("val_probs")
        val_targets = result.get("val_targets")
        if val_probs is not None and val_targets is not None:
            best_t, _ = find_best_threshold(val_targets, val_probs, threshold_search_steps)
        else:
            best_t = 0.5

        # Compute metrics
        metrics = compute_metrics(y_eval, test_probs, threshold=best_t)
        metrics["model"] = model_name

        # Lead-time analysis
        if ttp_eval is not None:
            lt = compute_lead_time(y_eval, test_probs, ttp_eval, threshold=best_t)
            metrics["mean_lead_time_min"] = lt["mean_lead_time"]
            metrics["median_lead_time_min"] = lt["median_lead_time"]
            metrics["n_true_positives"] = lt["n_true_positives"]
            metrics["lead_time_available"] = lt["available"]

            # Save lead times to CSV
            if lt["available"]:
                lt_df = pd.DataFrame({"lead_time_min": lt["lead_times"]})
                lt_df.to_csv(out_dir / f"lead_times_{model_name}.csv", index=False)
        else:
            metrics["mean_lead_time_min"] = float("nan")
            metrics["median_lead_time_min"] = float("nan")
            metrics["n_true_positives"] = 0
            metrics["lead_time_available"] = False

        # Save confusion matrix
        cm = metrics.pop("confusion_matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual_0", "Actual_1"],
            columns=["Pred_0", "Pred_1"],
        )
        cm_df.to_csv(out_dir / f"confusion_matrix_{model_name}.csv")

        all_metrics.append(metrics)
        logger.info(
            "  %s — F1=%.4f  PR-AUC=%.4f  ROC-AUC=%.4f  LeadTime=%.1f min",
            model_name,
            metrics["f1"],
            metrics["pr_auc"],
            metrics["roc_auc"],
            metrics["mean_lead_time_min"],
        )

    # Save summary
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(out_dir / "model_comparison.csv", index=False)
    logger.info("Evaluation summary saved to %s", out_dir / "model_comparison.csv")

    return summary_df


# ------------------------------------------------------------------ #
#  PR / ROC curve data (for visualization module)                      #
# ------------------------------------------------------------------ #

def get_pr_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for PR curve."""
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=float), nan=0.0)
    return precision_recall_curve(y_true, y_prob)


def get_roc_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for ROC curve."""
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=float), nan=0.0)
    return roc_curve(y_true, y_prob)
