"""
GCIS ML Pipeline — Isolation Forest + Rolling Z-Score Anomaly Detector

Always-on safety net, independent of the main LightGBM model.
Used as the fallback signal when the primary model is degraded.

Design:
  - sklearn IsolationForest trained on normal (success) transitions only.
    Contamination rate = IF_CONTAMINATION (default 5%).
  - Per-tag rolling z-score computed live from the TagBuffer.
  - Ensemble: anomaly is flagged if IF score OR z-score exceeds threshold.
  - Output: normalised anomaly_score per tag in [0, 1].
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from ..config import ALL_TAGS, IF_CONTAMINATION
from ..types import AnomalyReport
from datetime import datetime, timezone


# ─── Training ─────────────────────────────────────────────────────────────────


def train(
    X_normal: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Train Isolation Forest on normal (success) transitions.

    Args:
        X_normal: (n_samples, n_features) — features from successful transitions only
        output_dir: artefact directory

    Returns:
        metrics dict
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    model = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_normal)

    # Score distribution on training set
    scores = model.score_samples(X_normal)  # higher = more normal
    threshold = np.percentile(scores, IF_CONTAMINATION * 100)

    metrics = {
        "n_train_samples": len(X_normal),
        "n_features": X_normal.shape[1],
        "contamination": IF_CONTAMINATION,
        "score_threshold": float(threshold),
        "score_mean_normal": float(np.mean(scores)),
        "score_std_normal": float(np.std(scores)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_dir / "isolation_forest.pkl", "wb") as f:
        pickle.dump({"model": model, "threshold": threshold}, f)

    (output_dir / "feature_columns.txt").write_text(
        "\n".join(ALL_TAGS)
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[IsolationForest] Trained on {len(X_normal)} normal samples. threshold={threshold:.4f}")
    return metrics


# ─── Inference ────────────────────────────────────────────────────────────────


class AnomalyDetector:
    """
    Ensemble anomaly detector: Isolation Forest + per-tag rolling z-score.
    """

    # z-score threshold for flagging a tag
    Z_THRESHOLD: float = 3.0
    # history window for z-score (number of recent readings per tag)
    Z_WINDOW: int = 60  # 60 ticks = 5 minutes at 5s cadence

    def __init__(
        self,
        model: IsolationForest,
        threshold: float,
        feature_columns: list[str],
    ) -> None:
        self._model = model
        self._threshold = threshold
        self._feature_columns = feature_columns
        # Rolling z-score history per tag: deque of recent raw tag values
        from collections import deque
        self._tag_history: dict[str, "deque[float]"] = {
            tag: deque(maxlen=self.Z_WINDOW) for tag in ALL_TAGS
        }

    @classmethod
    def load(cls, model_dir: Path) -> "AnomalyDetector":
        with open(model_dir / "isolation_forest.pkl", "rb") as f:
            data = pickle.load(f)
        feature_columns = (
            (model_dir / "feature_columns.txt").read_text().strip().split("\n")
        )
        return cls(data["model"], data["threshold"], feature_columns)

    def push_reading(self, tag: str, value: float) -> None:
        """Update rolling history for a tag. Call on every new sensor reading."""
        if tag in self._tag_history:
            self._tag_history[tag].append(value)

    def _z_score_anomaly(self, tag: str, value: float) -> float:
        """Return normalised z-score-based anomaly score for a tag, in [0,1]."""
        history = list(self._tag_history.get(tag, []))
        if len(history) < 10:
            return 0.0
        mean = np.mean(history)
        std = np.std(history)
        if std < 1e-6:
            return 0.0
        z = abs((value - mean) / std)
        # Normalise: score = min(z / Z_THRESHOLD, 1)
        return float(np.clip(z / self.Z_THRESHOLD, 0.0, 1.0))

    def predict(
        self,
        tag_values: dict[str, float],
        feature_vector: dict[str, float] | None = None,
    ) -> AnomalyReport:
        """
        Compute anomaly report for the current tick.

        Args:
            tag_values: raw {tag_name: value} for the current tick
            feature_vector: optional engineered feature dict for IF scoring
        """
        # Per-tag z-score anomaly scores
        tag_scores: dict[str, float] = {}
        for tag, val in tag_values.items():
            self.push_reading(tag, val)
            tag_scores[tag] = self._z_score_anomaly(tag, val)

        # IF anomaly on feature vector if available
        if_anomaly = False
        if feature_vector is not None and self._feature_columns:
            X = np.array(
                [[feature_vector.get(col, 0.0) for col in self._feature_columns]]
            )
            score = float(self._model.score_samples(X)[0])
            if_anomaly = score < self._threshold
            # Add an "overall_if" key
            tag_scores["_isolation_forest"] = float(
                np.clip((self._threshold - score) / abs(self._threshold), 0.0, 1.0)
            )

        flagged = [
            tag
            for tag, score in tag_scores.items()
            if score >= 0.8 and not tag.startswith("_")
        ]
        overall = if_anomaly or any(
            s >= 1.0 for s in tag_scores.values()
        )

        return AnomalyReport(
            ts=datetime.now(timezone.utc),
            anomaly_scores=tag_scores,
            flagged_tags=flagged,
            overall_anomaly=overall,
        )
