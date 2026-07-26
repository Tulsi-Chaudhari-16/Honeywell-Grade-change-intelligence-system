"""
GCIS ML Pipeline — LightGBM Off-Spec Probability Model

Outputs:
  - p_offspec: probability that BW will exceed ±2.5% within the forecast horizon
  - Calibrated prediction intervals (quantile regression)
  - SHAP values per prediction for the Root Cause Agent

Design choices:
  - LightGBM over XGBoost for slightly faster training on sparse tabular data
    and cleaner SHAP integration via TreeExplainer
  - Three quantile models (q=0.05, 0.50, 0.95) alongside the main binary model
    for calibrated confidence intervals
  - SHAP TreeExplainer is deterministic and < 100ms for the feature count we have
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from ..config import ALL_TAGS, ROLLING_WINDOWS_SEC, ROLLING_STATS
from ..types import FactorAttribution, FeatureVector, Prediction, RootCauseAttribution


# ─── Feature column order (must be consistent across train and infer) ─────────


def _feature_columns(sample_features: dict[str, float]) -> list[str]:
    """Return sorted feature column names — order must be identical at train & infer."""
    return sorted(sample_features.keys())


# ─── Training ─────────────────────────────────────────────────────────────────


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Train the binary classifier + three quantile models + SHAP explainer.

    Args:
        X_train, y_train: training features and binary labels (1 = off-spec)
        X_val, y_val: held-out validation set
        output_dir: where to write model artefacts (picked up by registry)

    Returns:
        metrics dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Binary classifier ──────────────────────────────────────────────────────
    params_binary = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]

    model_binary = lgb.train(
        params_binary,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    # ── Quantile models for uncertainty intervals ──────────────────────────────
    quantile_models: dict[float, lgb.Booster] = {}
    for q in [0.05, 0.50, 0.95]:
        params_q = {
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
        }
        quantile_models[q] = lgb.train(
            params_q,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(200)],
        )

    # ── SHAP explainer ─────────────────────────────────────────────────────────
    explainer = shap.TreeExplainer(model_binary)

    # ── Validation metrics ─────────────────────────────────────────────────────
    from sklearn.metrics import average_precision_score, roc_auc_score

    preds = model_binary.predict(X_val)
    auc = roc_auc_score(y_val, preds)
    ap = average_precision_score(y_val, preds)

    # Brier score
    brier = float(np.mean((preds - y_val.values) ** 2))

    metrics = {
        "val_auc": round(auc, 4),
        "val_average_precision": round(ap, 4),
        "val_brier_score": round(brier, 4),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": X_train.shape[1],
        "best_iteration": model_binary.best_iteration,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"[LightGBM] AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}")

    # ── Persist artefacts ──────────────────────────────────────────────────────
    model_binary.save_model(str(output_dir / "lgbm_binary.txt"))
    with open(output_dir / "quantile_models.pkl", "wb") as f:
        pickle.dump(quantile_models, f)
    with open(output_dir / "shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    # Save feature column order
    (output_dir / "feature_columns.txt").write_text(
        "\n".join(X_train.columns.tolist())
    )

    import json
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


# ─── Inference ────────────────────────────────────────────────────────────────


class LGBMOffspecModel:
    """
    Loaded model for live inference. Call `load(model_dir)` to instantiate.
    Thread-safe for concurrent inference (LightGBM models are read-only after load).
    """

    def __init__(
        self,
        model: lgb.Booster,
        quantile_models: dict[float, lgb.Booster],
        explainer: shap.TreeExplainer,
        feature_columns: list[str],
    ) -> None:
        self._model = model
        self._quantile_models = quantile_models
        self._explainer = explainer
        self._feature_columns = feature_columns

    @classmethod
    def load(cls, model_dir: Path) -> "LGBMOffspecModel":
        model = lgb.Booster(model_file=str(model_dir / "lgbm_binary.txt"))
        with open(model_dir / "quantile_models.pkl", "rb") as f:
            quantile_models = pickle.load(f)
        with open(model_dir / "shap_explainer.pkl", "rb") as f:
            explainer = pickle.load(f)
        feature_columns = (model_dir / "feature_columns.txt").read_text().strip().split("\n")
        return cls(model, quantile_models, explainer, feature_columns)

    def _to_array(self, features: dict[str, float]) -> np.ndarray:
        """Convert feature dict to ordered numpy array matching training column order."""
        return np.array([[features.get(col, 0.0) for col in self._feature_columns]])

    def predict(self, feature_vector: FeatureVector) -> tuple[float, float, float]:
        """
        Returns:
            (p_offspec, lower_bound, upper_bound)
        """
        X = self._to_array(feature_vector.features)
        p = float(self._model.predict(X)[0])
        lower = float(self._quantile_models[0.05].predict(X)[0])
        upper = float(self._quantile_models[0.95].predict(X)[0])
        return np.clip(p, 0, 1), np.clip(lower, 0, 1), np.clip(upper, 0, 1)

    def explain(
        self,
        feature_vector: FeatureVector,
        prediction_id: str,
        top_k: int = 10,
    ) -> RootCauseAttribution:
        """
        Compute SHAP values and return a structured RootCauseAttribution.
        The Root Cause Agent (partner) narrates this — it never recomputes SHAP.
        """
        X = self._to_array(feature_vector.features)
        shap_values = self._explainer.shap_values(X)

        # For binary classifiers SHAP returns list [neg_class, pos_class]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        expected_value = float(
            self._explainer.expected_value[1]
            if isinstance(self._explainer.expected_value, (list, np.ndarray))
            else self._explainer.expected_value
        )

        # Rank by absolute SHAP magnitude
        ranked_idx = np.argsort(np.abs(sv))[::-1][:top_k]

        factors: list[FactorAttribution] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            col = self._feature_columns[idx]
            val = feature_vector.features.get(col, 0.0)
            sv_val = float(sv[idx])

            # Determine direction heuristic from feature name and SHAP sign
            if "rate_of_change" in col:
                direction = "too_fast" if sv_val > 0 else "too_slow"
            else:
                direction = "too_high" if sv_val > 0 else "too_low"

            factors.append(
                FactorAttribution(
                    feature_name=col,
                    shap_value=sv_val,
                    feature_value=val,
                    direction=direction,  # type: ignore[arg-type]
                    rank=rank,
                )
            )

        return RootCauseAttribution(
            prediction_id=prediction_id,
            episode_id=feature_vector.episode_id,
            ts=feature_vector.ts,
            ranked_factors=factors,
            attribution_method="shap",
            model_expected_value=expected_value,
        )

    def confidence_from_interval(self, lower: float, upper: float) -> float:
        """
        Model calibration confidence = 1 - normalised interval width.
        Interval width is normalised by dividing by the full [0,1] range.
        """
        width = upper - lower
        return float(np.clip(1.0 - width, 0.0, 1.0))
