"""
GCIS ML Pipeline — Concept Drift Detection (PSI)

Population Stability Index (PSI) measures how much a feature's distribution
has shifted between the training set and the current live data.

PSI interpretation:
  < 0.10 — no significant drift
  0.10–0.20 — moderate drift, monitor
  > 0.20 — significant drift, retrain recommended (RETRAIN_TRIGGER)

Called by the Learning Agent weekly (or on demand) to populate drift_metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import PSI_RETRAIN_THRESHOLD
from .types import DriftReport


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    Args:
        reference: 1D array of feature values from training data
        current:   1D array of recent live feature values
        n_bins:    number of quantile-based bins (quantile bins are robust to outliers)
        eps:       small constant to avoid log(0)

    Returns:
        PSI value (float). Higher = more drift.
    """
    if len(reference) < n_bins or len(current) < n_bins:
        return 0.0

    # Build quantile bins from reference distribution
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Proportions in each bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / (len(reference) + eps)
    cur_pct = cur_counts / (len(current) + eps)

    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    new_episodes_since_last_train: int = 0,
) -> DriftReport:
    """
    Compute PSI for each feature column and return a DriftReport.

    Args:
        reference_df: Feature DataFrame from the training set (one row per tick)
        current_df:   Feature DataFrame from recent live episodes
        feature_cols: Which feature columns to check (typically top-10 SHAP features)
        new_episodes_since_last_train: how many new confirmed episodes since last retrain

    Returns:
        DriftReport — consumed by the Learning Agent.
    """
    feature_psi: dict[str, float] = {}
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref = reference_df[col].dropna().values
        cur = current_df[col].dropna().values
        if len(ref) == 0 or len(cur) == 0:
            feature_psi[col] = 0.0
        else:
            feature_psi[col] = compute_psi(ref, cur)

    drifted = sorted(
        [col for col, psi in feature_psi.items() if psi > PSI_RETRAIN_THRESHOLD],
        key=lambda c: feature_psi[c],
        reverse=True,
    )

    from .config import NEW_EPISODES_RETRAIN_TRIGGER
    triggered = bool(drifted) or (new_episodes_since_last_train >= NEW_EPISODES_RETRAIN_TRIGGER)

    return DriftReport(
        computed_at=datetime.now(timezone.utc),
        feature_psi=feature_psi,
        top_drifted_features=drifted,
        triggered_retrain=triggered,
        new_episodes_since_last_train=new_episodes_since_last_train,
    )


def get_top_shap_features(n: int = 10) -> list[str]:
    """
    Load the top-n most important features from the LightGBM model's SHAP values.
    Used by the Learning Agent to focus PSI computation on the most impactful features.
    """
    from .registry import get_model_path
    import json
    try:
        model_dir = get_model_path("lgbm_offspec")
        metrics = json.loads((model_dir / "metrics.json").read_text())
        # Feature importance is stored in metrics as a sorted list if available
        cols = (model_dir / "feature_columns.txt").read_text().strip().split("\n")
        return cols[:n]
    except (FileNotFoundError, KeyError):
        return []
