"""
GCIS ML Pipeline — Train Isolation Forest Anomaly Detector

Trains on normal (success) transitions only.

Run from repo root:
    python -m backend.app.ml.train.train_isolation_forest
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.app.ml.config import ALL_TAGS, SEED_DATA_PATH
from backend.app.ml.features import extract_features_from_df
from backend.app.ml.models.isolation_forest import train as if_train
from backend.app.ml.registry import save_model


def main() -> None:
    print("=" * 60)
    print("GCIS — Training Isolation Forest Anomaly Detector")
    print("=" * 60)

    readings = pd.read_csv(SEED_DATA_PATH / "sensor_readings.csv", parse_dates=["ts"])
    transitions = pd.read_csv(SEED_DATA_PATH / "transitions.csv", parse_dates=["started_at", "ended_at"])

    from data.seed.generate_seed_data import GRADE_SPECS
    grade_codes = list(GRADE_SPECS.keys())
    recipes = pd.DataFrame(
        [{"grade_code": g, **{tag: GRADE_SPECS[g].get(tag, 0.0) for tag in ALL_TAGS}}
         for g in grade_codes]
    )

    # ── Use only successful transitions for IF training ────────────────────────
    success_transitions = transitions[transitions["outcome"] == "success"]
    print(f"Success transitions: {len(success_transitions)} / {len(transitions)}")

    features_df = extract_features_from_df(readings, success_transitions, recipes)
    feature_cols = [
        c for c in features_df.columns
        if c not in ("episode_id", "ts", "any_imputed")
    ]

    X = features_df[feature_cols].fillna(0.0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Feature matrix (normal only): {X_scaled.shape}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        import pickle
        with open(tmp_path / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        (tmp_path / "feature_columns.txt").write_text("\n".join(feature_cols))
        metrics = if_train(X_scaled, tmp_path)
        save_model("isolation_forest", "v1", tmp_path, metrics)

    print("\n[Success] Isolation Forest trained and saved to registry.")
    print(f"  Samples: {metrics['n_train_samples']}")
    print(f"  Score threshold (p5): {metrics['score_threshold']:.4f}")


if __name__ == "__main__":
    main()
