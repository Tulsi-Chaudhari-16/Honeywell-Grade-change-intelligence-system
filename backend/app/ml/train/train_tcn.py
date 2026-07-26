"""
GCIS ML Pipeline — Train TCN + LSTM Forecast Models

End-to-end:
  1. Load seed data
  2. Build sliding-window sequences per episode
  3. Train TCN (primary) + LSTM (ablation baseline)
  4. Compare RMSE — log if LSTM wins
  5. Save both to model registry

Run from repo root:
    python -m backend.app.ml.train.train_tcn
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
from sklearn.preprocessing import StandardScaler

from backend.app.ml.config import (
    ALL_TAGS,
    FORECAST_HORIZON_STEPS,
    INPUT_WINDOW_STEPS,
    SEED_DATA_PATH,
)
from backend.app.ml.features import extract_features_from_df
from backend.app.ml.models.tcn_forecast import build_sequences, train as tcn_train
from backend.app.ml.registry import save_model


def main() -> None:
    print("=" * 60)
    print("GCIS — Training TCN + LSTM Forecast Models")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    readings = pd.read_csv(SEED_DATA_PATH / "sensor_readings.csv", parse_dates=["ts"])
    transitions = pd.read_csv(SEED_DATA_PATH / "transitions.csv", parse_dates=["started_at", "ended_at"])

    from data.seed.generate_seed_data import GRADE_SPECS
    grade_codes = list(GRADE_SPECS.keys())
    recipes = pd.DataFrame(
        [{"grade_code": g, **{tag: GRADE_SPECS[g].get(tag, 0.0) for tag in ALL_TAGS}}
         for g in grade_codes]
    )

    print(f"Loaded: {len(transitions)} transitions, {len(readings)} readings")

    # ── Extract features ───────────────────────────────────────────────────────
    print("\nExtracting features...")
    features_df = extract_features_from_df(readings, transitions, recipes)

    # BW series (target)
    bw_readings = readings[readings["tag_name"] == "basis_weight"][["episode_id", "ts", "value"]]
    bw_readings = bw_readings.rename(columns={"value": "basis_weight"})
    features_df["ts"] = pd.to_datetime(features_df["ts"])
    bw_readings["ts"] = pd.to_datetime(bw_readings["ts"])
    merged = features_df.merge(bw_readings, on=["episode_id", "ts"], how="inner")

    feature_cols = [
        c for c in merged.columns
        if c not in ("episode_id", "ts", "basis_weight", "any_imputed")
    ]

    # ── Build per-episode sequences ────────────────────────────────────────────
    print("\nBuilding sliding-window sequences...")
    all_X, all_y, all_groups = [], [], []

    # Normalise features across all episodes first
    scaler = StandardScaler()
    merged[feature_cols] = scaler.fit_transform(merged[feature_cols].fillna(0.0))

    for ep_id, ep_df in merged.groupby("episode_id"):
        ep_df = ep_df.sort_values("ts")
        if len(ep_df) < INPUT_WINDOW_STEPS + FORECAST_HORIZON_STEPS:
            continue
        X_ep, y_ep = build_sequences(ep_df[feature_cols], ep_df["basis_weight"])
        all_X.append(X_ep)
        all_y.append(y_ep)
        all_groups.extend([ep_id] * len(X_ep))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.array(all_groups)
    print(f"Total sequences: {len(X)} | Features: {X.shape[1]} | Horizon: {y.shape[1]}")

    # ── Group-aware split ──────────────────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"Train seqs: {len(X_train)} | Val seqs: {len(X_val)}")

    # ── Train ──────────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Save scaler alongside model
        import pickle
        with open(tmp_path / "feature_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        (tmp_path / "feature_columns.txt").write_text("\n".join(feature_cols))

        metrics = tcn_train(X_train, y_train, X_val, y_val, tmp_path)
        save_model("tcn_forecast", "v1", tmp_path, metrics)

    print("\n[Success] TCN + LSTM trained and saved to registry.")
    print(f"  TCN  val RMSE:  {metrics['tcn_val_rmse']}")
    print(f"  LSTM val RMSE:  {metrics['lstm_val_rmse']}")
    print(f"  Naive baseline: {metrics['naive_baseline_rmse']}")
    if metrics["tcn_vs_lstm_improvement_pct"] < 0:
        print("  [WARNING] LSTM outperformed TCN — review training data or hyperparameters.")
    else:
        print(f"  TCN improvement over LSTM: {metrics['tcn_vs_lstm_improvement_pct']}%")


if __name__ == "__main__":
    main()
