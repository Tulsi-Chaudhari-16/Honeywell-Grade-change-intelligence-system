"""
GCIS ML Pipeline — Train LightGBM Off-Spec Model

End-to-end:
  1. Load seed data (transitions + sensor readings + labels)
  2. Compute features via features.py (batch mode)
  3. Train LightGBM binary + quantile models + SHAP explainer
  4. Backtest on held-out transitions
  5. Save to model registry

Run from repo root:
    python -m backend.app.ml.train.train_lgbm
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from backend.app.ml.config import SEED_DATA_PATH, ALL_TAGS
from backend.app.ml.features import (
    compute_and_save_envelope,
    extract_features_from_df,
)
from backend.app.ml.models.lgbm_offspec import train as lgbm_train
from backend.app.ml.registry import save_model


def _load_seed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    readings = pd.read_csv(SEED_DATA_PATH / "sensor_readings.csv", parse_dates=["ts"])
    transitions = pd.read_csv(SEED_DATA_PATH / "transitions.csv", parse_dates=["started_at", "ended_at"])
    labels = pd.read_csv(SEED_DATA_PATH / "training_examples.csv", parse_dates=["ts"])

    # Simple grade recipes derived from GRADE_SPECS (subset of tags used as setpoints)
    from backend.app.ml.config import GRADE_PAIRS
    from data.seed.generate_seed_data import GRADE_SPECS

    grade_codes = list(GRADE_SPECS.keys())
    recipes = pd.DataFrame(
        [{"grade_code": g, **{tag: GRADE_SPECS[g].get(tag, 0.0) for tag in ALL_TAGS}}
         for g in grade_codes]
    )
    return readings, transitions, labels, recipes


def main() -> None:
    print("=" * 60)
    print("GCIS — Training LightGBM Off-Spec Model")
    print("=" * 60)

    readings, transitions, labels, recipes = _load_seed_data()
    print(f"Loaded: {len(transitions)} transitions, {len(readings)} readings, {len(labels)} labels")

    # ── Compute envelopes for each grade pair ──────────────────────────────────
    print("\nComputing historical envelopes...")
    for from_g, to_g in transitions[["from_grade", "to_grade"]].drop_duplicates().values:
        grade_pair = f"{from_g}__{to_g}"
        success_eps = transitions[
            (transitions["from_grade"] == from_g)
            & (transitions["to_grade"] == to_g)
            & (transitions["outcome"] == "success")
        ]["episode_id"].tolist()

        if not success_eps:
            continue

        ep_readings = readings[readings["episode_id"].isin(success_eps)].copy()
        # Pivot to wide format for envelope computation
        wide = ep_readings.pivot_table(
            index=["episode_id", "ts"], columns="tag_name", values="value"
        ).reset_index()
        wide["step"] = wide.groupby("episode_id").cumcount()
        compute_and_save_envelope(wide, grade_pair)

    # ── Batch feature extraction ───────────────────────────────────────────────
    print("\nExtracting features (this may take a few minutes)...")
    features_df = extract_features_from_df(readings, transitions, recipes)
    print(f"Feature matrix: {features_df.shape}")

    # ── Join with labels ───────────────────────────────────────────────────────
    labels["ts"] = pd.to_datetime(labels["ts"])
    features_df["ts"] = pd.to_datetime(features_df["ts"])
    merged = features_df.merge(
        labels[["episode_id", "ts", "label_offspec"]],
        on=["episode_id", "ts"],
        how="inner",
    ).dropna(subset=["label_offspec"])

    print(f"Merged dataset: {len(merged)} samples, "
          f"offspec rate={merged['label_offspec'].mean():.2%}")

    feature_cols = [c for c in merged.columns if c not in ("episode_id", "ts", "label_offspec", "any_imputed")]
    X = merged[feature_cols].fillna(0.0)
    y = merged["label_offspec"].astype(int)
    groups = merged["episode_id"]  # group-aware split: whole episodes go to train or val

    # ── Group-aware train/val split (no episode leakage) ──────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # ── Train ──────────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        metrics = lgbm_train(X_train, y_train, X_val, y_val, tmp_path)
        version = "v1"
        save_model("lgbm_offspec", version, tmp_path, metrics)

    print("\n[Success] LightGBM model trained and saved to registry.")
    print(f"  AUC:               {metrics['val_auc']}")
    print(f"  Average Precision: {metrics['val_average_precision']}")
    print(f"  Brier Score:       {metrics['val_brier_score']}")


if __name__ == "__main__":
    main()
