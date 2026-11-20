"""
Anomaly detection models for cryptocurrency pump-and-dump detection.
"""
import logging
import os
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


def handle_nans(X: pd.DataFrame) -> pd.DataFrame:
    """Handle NaN and inf values by replacing with 0 and logging a warning."""
    if X.isnull().values.any() or np.isinf(X.values).any():
        logger.warning("NaN or inf values found in features. Filling with 0.")
        return X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def train_isolation_forest(X_train: pd.DataFrame, X_val: pd.DataFrame, 
                           X_test: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Train an Isolation Forest model for anomaly detection.
    
    Converts anomaly scores to probability-like values in [0, 1].
    Higher score indicates higher probability of anomaly (pump).
    """
    logger.info("Training Isolation Forest...")
    
    X_train_clean = handle_nans(X_train)
    X_val_clean = handle_nans(X_val)
    X_test_clean = handle_nans(X_test)
    
    random_seed = cfg.get("random_seed", 42)
    iso_cfg = cfg.get("isolation_forest", {})
    
    model = IsolationForest(
        n_estimators=iso_cfg.get("n_estimators", 100),
        contamination=iso_cfg.get("contamination", "auto"),
        max_samples=iso_cfg.get("max_samples", "auto"),
        random_state=random_seed
    )
    
    model.fit(X_train_clean)
    
    models_dir = cfg.get("data", {}).get("models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "isolation_forest_model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Isolation Forest model saved to {model_path}")
    
    def get_probs(X_data: pd.DataFrame) -> np.ndarray:
        if X_data is None or len(X_data) == 0:
            return None
        # decision_function returns lower values for anomalies.
        # We negate it so higher values mean more anomalous.
        scores = -model.decision_function(X_data)
        
        # Normalize to [0, 1] using min-max scaling (rough approximation of prob)
        s_min = scores.min()
        s_max = scores.max()
        if s_max > s_min:
            probs = (scores - s_min) / (s_max - s_min)
        else:
            probs = np.zeros_like(scores)
        return probs
        
    val_probs = get_probs(X_val_clean)
    test_probs = get_probs(X_test_clean)
    
    return model, val_probs, test_probs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    X_tr = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
    X_v = pd.DataFrame(np.random.randn(20, 3), columns=["f1", "f2", "f3"])
    X_te = pd.DataFrame(np.random.randn(20, 3), columns=["f1", "f2", "f3"])
    
    dummy_cfg = {
        "random_seed": 42,
        "project_root": ".",
        "isolation_forest": {
            "n_estimators": 50,
            "contamination": 0.05
        }
    }
    
    model, v_probs, t_probs = train_isolation_forest(X_tr, X_v, X_te, dummy_cfg)
    print(f"Val probs mean: {v_probs.mean():.4f}")
