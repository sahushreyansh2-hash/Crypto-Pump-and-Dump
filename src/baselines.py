"""
Baseline models for cryptocurrency pump-and-dump detection.
"""
import logging
import os
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)

def handle_nans(X: pd.DataFrame) -> pd.DataFrame:
    """Handle NaN and inf values by replacing with 0 and logging a warning."""
    if X.isnull().values.any() or np.isinf(X.values).any():
        logger.warning("NaN or inf values found in features. Filling with 0.")
        return X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


class RuleBasedBaseline:
    """Rule-based model for pump-and-dump detection."""
    def __init__(self, volume_z_score_threshold: float, price_return_threshold: float, volatility_threshold: float):
        self.volume_z_score_threshold = volume_z_score_threshold
        self.price_return_threshold = price_return_threshold
        self.volatility_threshold = volatility_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit method (no-op for rule-based)."""
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability-like scores based on rules."""
        X_clean = handle_nans(X)
        
        vol_z = X_clean.get("vol_zscore", pd.Series(np.zeros(len(X_clean)), index=X_clean.index))
        price_ret = X_clean.get("pct_return", pd.Series(np.zeros(len(X_clean)), index=X_clean.index))
        volatility = X_clean.get("volatility_10", pd.Series(np.zeros(len(X_clean)), index=X_clean.index))

        # 1 if all conditions met, else 0
        preds = (
            (vol_z > self.volume_z_score_threshold) &
            (price_ret > self.price_return_threshold) &
            (volatility > self.volatility_threshold)
        ).astype(int)
        
        # Format as [prob_class_0, prob_class_1]
        probs = np.zeros((len(X_clean), 2))
        probs[:, 1] = preds.values
        probs[:, 0] = 1 - preds.values
        return probs

def train_and_predict(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame, X_test: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Train a specific baseline model and return predictions."""
    logger.info(f"Training model: {model_name}")
    X_train_clean = handle_nans(X_train)
    X_val_clean = handle_nans(X_val)
    X_test_clean = handle_nans(X_test)
    
    random_seed = cfg.get("random_seed", 42)
    models_dir = cfg.get("data", {}).get("models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model = None
    if model_name == "rule_based":
        rb_cfg = cfg.get("rule_based", {})
        model = RuleBasedBaseline(
            volume_z_score_threshold=rb_cfg.get("volume_z_score_threshold", 3.0),
            price_return_threshold=rb_cfg.get("price_return_threshold", 0.05),
            volatility_threshold=rb_cfg.get("volatility_threshold", 0.02)
        )
        model.fit(X_train_clean, y_train)
        
    elif model_name == "logistic_regression":
        lr_cfg = cfg.get("logistic_regression", {})
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=lr_cfg.get("max_iter", 1000),
            random_state=random_seed
        )
        model.fit(X_train_clean, y_train)
        
    elif model_name == "random_forest":
        rf_cfg = cfg.get("random_forest", {})
        model = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=rf_cfg.get("n_estimators", 100),
            max_depth=rf_cfg.get("max_depth", None),
            random_state=random_seed
        )
        model.fit(X_train_clean, y_train)
        
    elif model_name == "xgboost":
        if not XGB_AVAILABLE:
            logger.warning("XGBoost is not available. Skipping xgboost model.")
            return {"model": None, "val_probs": None, "test_probs": None}
            
        xgb_cfg = cfg.get("xgboost", {})
        scale_pos_weight = xgb_cfg.get("scale_pos_weight", None)
        
        if scale_pos_weight is None:
            num_neg = (y_train == 0).sum()
            num_pos = (y_train == 1).sum()
            scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
            
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            random_state=random_seed,
            n_estimators=xgb_cfg.get("n_estimators", 100),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.1)
        )
        model.fit(X_train_clean, y_train)
        
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Save model
    model_path = os.path.join(models_dir, f"{model_name}_model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model {model_name} saved to {model_path}")

    val_probs = model.predict_proba(X_val_clean)[:, 1] if X_val_clean is not None and len(X_val_clean) > 0 else None
    test_probs = model.predict_proba(X_test_clean)[:, 1] if X_test_clean is not None and len(X_test_clean) > 0 else None

    return {
        "model": model,
        "val_probs": val_probs,
        "test_probs": test_probs
    }


def run_all_baselines(X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame, X_test: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Run all baseline models and return results."""
    results = {}
    models_to_run = ["rule_based", "logistic_regression", "random_forest", "xgboost"]
    
    for model_name in models_to_run:
        results[model_name] = train_and_predict(model_name, X_train, y_train, X_val, X_test, cfg)
        
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Dummy data
    X_tr = pd.DataFrame(np.random.randn(100, 3), columns=["volume_z_score", "price_return", "volatility"])
    y_tr = pd.Series(np.random.randint(0, 2, 100))
    X_v = pd.DataFrame(np.random.randn(20, 3), columns=["volume_z_score", "price_return", "volatility"])
    X_te = pd.DataFrame(np.random.randn(20, 3), columns=["volume_z_score", "price_return", "volatility"])
    
    dummy_cfg = {
        "random_seed": 42,
        "project_root": ".",
        "rule_based": {
            "volume_z_score_threshold": 1.0,
            "price_return_threshold": 0.1,
            "volatility_threshold": 0.1
        },
        "logistic_regression": {"max_iter": 100},
        "random_forest": {"n_estimators": 10},
        "xgboost": {"n_estimators": 10}
    }
    
    res = run_all_baselines(X_tr, y_tr, X_v, X_te, dummy_cfg)
    for m, r in res.items():
        if r["model"] is not None:
            print(f"{m} val probs shape: {r['val_probs'].shape}")
