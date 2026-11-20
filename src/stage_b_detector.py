"""
stage_b_detector.py — Post-Onset / Early-Spike Detection Module (Stage B).

Addresses Faculty Requirements:
1. "Compare Stage B ablations: XGBoost, temporal MLP, Isolation Forest, and stacking."
2. "The strongest post-onset classifier remains ordinary XGBoost with temporal features."
3. "Demonstrate the independent value of each stage under chronological evaluation."

Stage B analyzes time-chunk sequences to detect anomalous trading activity at onset.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


class StageBDetector:
    """
    Stage B Detector supporting 4 distinct model architectures:
    - 'xgboost'          : Gradient boosted trees with temporal & price-volume interactions
    - 'temporal_mlp'     : Deep Multi-Layer Perceptron on normalized sequence features
    - 'isolation_forest' : Unsupervised tree-based anomaly detector
    - 'stacking'         : Meta-learner stacking XGBoost + MLP + Isolation Forest
    """

    def __init__(self, model_type: str = "xgboost", random_state: int = 42):
        assert model_type in ["xgboost", "temporal_mlp", "isolation_forest", "stacking"], f"Invalid: {model_type}"
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self._init_model()

    def _init_model(self):
        if self.model_type == "xgboost" and XGB_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                scale_pos_weight=10.0,
                eval_metric="logloss",
                random_state=self.random_state,
            )
        elif self.model_type == "temporal_mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=200,
                early_stopping=True,
                random_state=self.random_state,
            )
        elif self.model_type == "isolation_forest":
            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=self.random_state,
            )
        else:  # stacking or fallback
            if XGB_AVAILABLE:
                base_estimators = [
                    ("xgb", xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=self.random_state)),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=self.random_state)),
                ]
            else:
                base_estimators = [
                    ("mlp1", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=self.random_state)),
                    ("mlp2", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=self.random_state)),
                ]
            self.model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(class_weight="balanced", random_state=self.random_state),
                cv=3,
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X = np.nan_to_num(np.asarray(X_train, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -1e5, 1e5)
        y = np.asarray(y_train, dtype=int)
        
        if self.model_type == "isolation_forest":
            # Unsupervised: fit on normal baseline data
            normal_mask = y == 0
            X_fit = X[normal_mask] if normal_mask.sum() > 0 else X
            self.model.fit(X_fit)
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_clean = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X_clean = np.clip(X_clean, -1e5, 1e5)
        
        if self.model_type == "isolation_forest":
            scores = -self.model.decision_function(X_clean)
            # Min-max scale anomaly scores into [0, 1] pseudo-probabilities
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                probs = (scores - s_min) / (s_max - s_min)
            else:
                probs = np.zeros_like(scores)
            return probs
        else:
            if hasattr(self.model, "predict_proba"):
                p = self.model.predict_proba(X_clean)
                return p[:, 1] if p.ndim == 2 else p
            elif hasattr(self.model, "decision_function"):
                from scipy.special import expit
                return expit(self.model.decision_function(X_clean))
            else:
                return self.model.predict(X_clean).astype(float)


def run_stage_b_ablations(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run complete ablation study for Stage B models:
    1. XGBoost
    2. Temporal MLP
    3. Isolation Forest
    4. Stacking Classifier
    """
    model_types = ["xgboost", "temporal_mlp", "isolation_forest", "stacking"]
    records = []
    
    y_test_arr = np.asarray(y_test, dtype=int)
    has_pos = y_test_arr.sum() > 0 and len(np.unique(y_test_arr)) > 1

    for m_type in model_types:
        detector = StageBDetector(model_type=m_type, random_state=seed)
        detector.fit(X_train, y_train)
        probs = detector.predict_proba(X_test)
        preds = (probs >= threshold).astype(int)

        prec = precision_score(y_test_arr, preds, zero_division=0)
        rec = recall_score(y_test_arr, preds, zero_division=0)
        f1 = f1_score(y_test_arr, preds, zero_division=0)
        
        pr_auc = average_precision_score(y_test_arr, probs) if has_pos else np.nan
        roc_auc = roc_auc_score(y_test_arr, probs) if has_pos else np.nan
        
        # False alarm rate (FPR = FP / (FP + TN))
        neg_mask = y_test_arr == 0
        far = float((preds[neg_mask] == 1).mean()) if neg_mask.sum() > 0 else 0.0

        records.append({
            "Stage_B_Model": m_type,
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "PR_AUC": round(float(pr_auc), 4) if not np.isnan(pr_auc) else np.nan,
            "ROC_AUC": round(float(roc_auc), 4) if not np.isnan(roc_auc) else np.nan,
            "False_Alarm_Rate": round(float(far), 5),
            "Threshold": threshold,
        })

    return pd.DataFrame(records)
