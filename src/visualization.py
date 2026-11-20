"""
visualization.py — Generate all plots from actual experiment results.

Plots generated
---------------
1.  Price and volume chart
2.  Volume z-score chart
3.  Detected pump events overlay
4.  Prediction probabilities over time
5.  Precision-Recall curve
6.  ROC curve
7.  Confusion matrix heatmap
8.  Model comparison bar chart
9.  Lead-time distribution histogram
10. False-alarm comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.evaluation import get_pr_curve_data, get_roc_curve_data

logger = logging.getLogger(__name__)

# Global style
sns.set_theme(style="whitegrid")
FIGSIZE = (12, 6)
DPI = 150


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved plot: %s", path)


# ================================================================== #
#  1. Price & Volume                                                   #
# ================================================================== #

def plot_price_volume(
    df: pd.DataFrame,
    symbol: str,
    output_dir: str = "results",
) -> None:
    """Dual-axis chart: close price + volume bars."""
    sub = df[df["symbol"] == symbol].copy()
    if sub.empty:
        logger.warning("No data for symbol %s", symbol)
        return

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax1.plot(sub["timestamp"], sub["close"], color="steelblue", linewidth=0.8, label="Close")
    ax1.set_ylabel("Close Price", color="steelblue")
    ax1.set_xlabel("Time")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.bar(sub["timestamp"], sub["volume"], alpha=0.3, color="orange", width=0.001, label="Volume")
    ax2.set_ylabel("Volume", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    fig.suptitle(f"Price & Volume — {symbol}", fontsize=14)
    fig.tight_layout()
    sym_clean = symbol.replace("/", "_")
    _save(fig, Path(output_dir) / f"price_volume_{sym_clean}.png")


# ================================================================== #
#  2. Volume Z-Score                                                   #
# ================================================================== #

def plot_volume_zscore(
    df: pd.DataFrame,
    symbol: str,
    threshold: float = 3.0,
    output_dir: str = "results",
) -> None:
    """Volume z-score with threshold line."""
    sub = df[df["symbol"] == symbol].copy()
    if sub.empty or "vol_zscore" not in sub.columns:
        logger.warning("Cannot plot volume z-score for %s", symbol)
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sub["timestamp"], sub["vol_zscore"], linewidth=0.7, color="purple")
    ax.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")
    ax.axhline(y=-threshold, color="red", linestyle="--")
    ax.set_title(f"Volume Z-Score — {symbol}")
    ax.set_ylabel("Z-Score")
    ax.legend()
    fig.tight_layout()
    sym_clean = symbol.replace("/", "_")
    _save(fig, Path(output_dir) / f"volume_zscore_{sym_clean}.png")


# ================================================================== #
#  3. Detected Pump Events                                             #
# ================================================================== #

def plot_pump_events(
    df: pd.DataFrame,
    pump_events: pd.DataFrame,
    symbol: str,
    output_dir: str = "results",
) -> None:
    """Price chart with vertical lines at known pump events."""
    sub = df[df["symbol"] == symbol].copy()
    events = pump_events[pump_events["symbol"] == symbol]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sub["timestamp"], sub["close"], linewidth=0.8, color="steelblue")

    for _, evt in events.iterrows():
        ax.axvline(x=evt["event_timestamp"], color="red", alpha=0.7, linestyle="--", linewidth=1.2)

    ax.set_title(f"Price with Known Pump Events — {symbol}")
    ax.set_ylabel("Close Price")
    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", label="Close Price"),
        Line2D([0], [0], color="red", linestyle="--", label="Pump Event"),
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    sym_clean = symbol.replace("/", "_")
    _save(fig, Path(output_dir) / f"pump_events_{sym_clean}.png")


# ================================================================== #
#  4. Prediction Probabilities                                         #
# ================================================================== #

def plot_prediction_probs(
    timestamps: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
    output_dir: str = "results",
) -> None:
    """Prediction probability over time with true labels highlighted."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(timestamps, probs, linewidth=0.6, color="steelblue", label="Pred Prob")
    ax.axhline(y=threshold, color="gray", linestyle="--", alpha=0.5, label=f"Threshold ({threshold:.2f})")

    # Shade positive regions
    pos_mask = y_true == 1
    if pos_mask.any():
        ax.fill_between(
            timestamps, 0, 1, where=pos_mask,
            alpha=0.15, color="red", label="Actual Pump Window",
        )

    ax.set_title(f"Prediction Probabilities — {model_name}")
    ax.set_ylabel("Pump Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, Path(output_dir) / f"pred_probs_{model_name}.png")


# ================================================================== #
#  5. Precision-Recall Curve                                           #
# ================================================================== #

def plot_pr_curve(
    model_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str = "results",
) -> None:
    """
    PR curve for one or more models.

    Parameters
    ----------
    model_curves : dict
        model_name -> (y_true, y_prob)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (y_true, y_prob) in model_curves.items():
        prec, rec, _ = get_pr_curve_data(y_true, y_prob)
        from sklearn.metrics import auc as sk_auc
        pr_auc = sk_auc(rec, prec)
        ax.plot(rec, prec, linewidth=1.2, label=f"{name} (AUC={pr_auc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _save(fig, Path(output_dir) / "pr_curve.png")


# ================================================================== #
#  6. ROC Curve                                                        #
# ================================================================== #

def plot_roc_curve(
    model_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str = "results",
) -> None:
    """ROC curve for one or more models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    for name, (y_true, y_prob) in model_curves.items():
        fpr, tpr, _ = get_roc_curve_data(y_true, y_prob)
        from sklearn.metrics import auc as sk_auc
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, Path(output_dir) / "roc_curve.png")


# ================================================================== #
#  7. Confusion Matrix                                                 #
# ================================================================== #

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str = "results",
) -> None:
    """Heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Pump"],
        yticklabels=["Normal", "Pump"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    _save(fig, Path(output_dir) / f"confusion_matrix_{model_name}.png")


# ================================================================== #
#  8. Model Comparison                                                 #
# ================================================================== #

def plot_model_comparison(
    summary_df: pd.DataFrame,
    output_dir: str = "results",
) -> None:
    """Grouped bar chart comparing F1, PR-AUC, ROC-AUC across models."""
    metrics = ["f1", "pr_auc", "roc_auc"]
    present = [m for m in metrics if m in summary_df.columns]

    melted = summary_df.melt(
        id_vars="model", value_vars=present, var_name="metric", value_name="score"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    ax.legend(title="Metric")
    fig.tight_layout()
    _save(fig, Path(output_dir) / "model_comparison.png")


# ================================================================== #
#  9. Lead-Time Distribution                                           #
# ================================================================== #

def plot_lead_time_distribution(
    lead_times: Dict[str, List[float]],
    output_dir: str = "results",
) -> None:
    """Histogram of lead times (minutes before pump) for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, times in lead_times.items():
        if times:
            ax.hist(times, bins=20, alpha=0.5, label=name, edgecolor="black")

    if not any(lead_times.values()):
        ax.text(
            0.5, 0.5,
            "No true-positive predictions — lead-time unavailable",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )

    ax.set_xlabel("Lead Time (minutes before pump)")
    ax.set_ylabel("Count")
    ax.set_title("Lead-Time Distribution")
    ax.legend()
    fig.tight_layout()
    _save(fig, Path(output_dir) / "lead_time_distribution.png")


# ================================================================== #
# 10. False Alarm Comparison                                           #
# ================================================================== #

def plot_false_alarm_comparison(
    summary_df: pd.DataFrame,
    output_dir: str = "results",
) -> None:
    """Bar chart comparing false alarm rates across models."""
    if "false_alarm_rate" not in summary_df.columns:
        logger.warning("No false_alarm_rate in summary — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary_df, x="model", y="false_alarm_rate", color="salmon", ax=ax)
    ax.set_title("False Alarm Rate Comparison")
    ax.set_ylabel("False Alarm Rate (FP / (FP + TN))")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, Path(output_dir) / "false_alarm_comparison.png")


# ================================================================== #
#  Master plot runner                                                  #
# ================================================================== #

def generate_all_plots(
    df: pd.DataFrame,
    pump_events: pd.DataFrame,
    model_results: Dict[str, Dict],
    summary_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
) -> None:
    """
    Generate all plots from actual experiment results.

    Parameters
    ----------
    df : pd.DataFrame
        Full featured dataframe (for price/volume/zscore plots).
    pump_events : pd.DataFrame
        Known pump events.
    model_results : dict
        model_name -> {test_probs, val_probs, ...}
    summary_df : pd.DataFrame
        Output of evaluate_all_models.
    test_df : pd.DataFrame
        Test set dataframe with timestamp, target, time_to_next_pump_min.
    cfg : dict
        Config dictionary.
    """
    output_dir = cfg["data"].get("results_dir", "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Generating plots …")

    # Per-symbol plots
    for symbol in df["symbol"].unique():
        plot_price_volume(df, symbol, output_dir)
        vol_t = cfg.get("rule_based", {}).get("volume_zscore_threshold", 3.0)
        plot_volume_zscore(df, symbol, threshold=vol_t, output_dir=output_dir)
        plot_pump_events(df, pump_events, symbol, output_dir)

    # Curves (PR, ROC)
    model_curves = {}
    lead_times_dict = {}
    y_test = test_df["target"].values

    for name, res in model_results.items():
        tp = res.get("test_probs")
        if tp is None:
            continue

        # Align lengths
        y_eval = y_test
        if len(tp) < len(y_test):
            y_eval = y_test[len(y_test) - len(tp):]

        model_curves[name] = (y_eval, tp)

        # Prediction probability chart
        ts = test_df["timestamp"].values
        if len(tp) < len(ts):
            ts = ts[len(ts) - len(tp):]
        try:
            threshold = summary_df.loc[summary_df["model"] == name, "threshold"].values[0]
        except (IndexError, KeyError):
            threshold = 0.5
        plot_prediction_probs(ts, tp, y_eval, name, threshold=threshold, output_dir=output_dir)

        # Confusion matrix
        y_pred = (tp >= threshold).astype(int)
        plot_confusion_matrix(y_eval, y_pred, name, output_dir)

        # Lead times
        lt_path = Path(output_dir) / f"lead_times_{name}.csv"
        if lt_path.exists():
            lt_data = pd.read_csv(lt_path)["lead_time_min"].tolist()
            lead_times_dict[name] = lt_data
        else:
            lead_times_dict[name] = []

    if model_curves:
        plot_pr_curve(model_curves, output_dir)
        plot_roc_curve(model_curves, output_dir)

    plot_model_comparison(summary_df, output_dir)
    plot_lead_time_distribution(lead_times_dict, output_dir)
    plot_false_alarm_comparison(summary_df, output_dir)

    logger.info("All plots generated.")
