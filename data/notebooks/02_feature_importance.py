"""
GCIS — Feature Importance Analysis

Loads the trained LightGBM model and produces:
  - Global SHAP summary plot (beeswarm)
  - Top-20 feature importance bar chart
  - SHAP values vs feature value scatter for top 5 features

Run AFTER training: python data/notebooks/02_feature_importance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from backend.app.ml.config import ALL_TAGS, SEED_DATA_PATH
from backend.app.ml.models.lgbm_offspec import LGBMOffspecModel
from backend.app.ml.registry import get_model_path
from backend.app.ml.features import extract_features_from_df

plt.style.use("dark_background")

OUTPUT_DIR = SEED_DATA_PATH.parent / "notebooks" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ── Load model ────────────────────────────────────────────────────────────
    model_dir = get_model_path("lgbm_offspec")
    lgbm = LGBMOffspecModel.load(model_dir)
    print(f"Loaded LightGBM from {model_dir}")

    # ── Load a sample of validation data ─────────────────────────────────────
    transitions = pd.read_csv(SEED_DATA_PATH / "transitions.csv", parse_dates=["started_at", "ended_at"])
    readings = pd.read_csv(SEED_DATA_PATH / "sensor_readings.csv", parse_dates=["ts"])
    labels = pd.read_csv(SEED_DATA_PATH / "training_examples.csv", parse_dates=["ts"])

    from data.seed.generate_seed_data import GRADE_SPECS
    grade_codes = list(GRADE_SPECS.keys())
    recipes = pd.DataFrame(
        [{"grade_code": g, **{tag: GRADE_SPECS[g].get(tag, 0.0) for tag in ALL_TAGS}}
         for g in grade_codes]
    )

    print("Extracting features for SHAP analysis (sampling 50 episodes)...")
    sample_eps = transitions["episode_id"].sample(min(50, len(transitions)), random_state=42)
    small_transitions = transitions[transitions["episode_id"].isin(sample_eps)]
    small_readings = readings[readings["episode_id"].isin(sample_eps)]
    features_df = extract_features_from_df(small_readings, small_transitions, recipes)

    feature_cols = [
        c for c in features_df.columns
        if c not in ("episode_id", "ts", "any_imputed")
    ]
    X_sample = features_df[feature_cols].fillna(0.0)
    # Reindex to model's expected column order
    X_sample = X_sample.reindex(columns=lgbm._feature_columns, fill_value=0.0)

    # ── SHAP values ───────────────────────────────────────────────────────────
    print("Computing SHAP values...")
    shap_values = lgbm._explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        sv = shap_values[1]   # positive class
    else:
        sv = shap_values

    # ── 1. Top-20 global feature importance bar chart ─────────────────────────
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top20_idx = np.argsort(mean_abs_shap)[::-1][:20]
    top20_names = [lgbm._feature_columns[i] for i in top20_idx]
    top20_vals = mean_abs_shap[top20_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(20), top20_vals[::-1], color="#f59e0b")
    ax.set_yticks(range(20))
    ax.set_yticklabels([n.replace("_", " ") for n in top20_names[::-1]], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Features — Global SHAP Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_top20_shap.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> {OUTPUT_DIR / '04_top20_shap.png'}")

    # ── 2. SHAP summary (beeswarm) via shap library ───────────────────────────
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv,
        X_sample,
        feature_names=lgbm._feature_columns,
        max_display=15,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {OUTPUT_DIR / '05_shap_beeswarm.png'}")

    # ── 3. SHAP value vs feature value for top 5 ──────────────────────────────
    top5_idx = np.argsort(mean_abs_shap)[::-1][:5]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("SHAP vs Feature Value — Top 5 Features", fontsize=12)
    for j, (idx, ax) in enumerate(zip(top5_idx, axes)):
        col = lgbm._feature_columns[idx]
        ax.scatter(
            X_sample.iloc[:, idx], sv[:, idx],
            c="#f59e0b", alpha=0.5, s=10
        )
        ax.axhline(0, color="white", linestyle="--", linewidth=0.5)
        ax.set_title(col.replace("_", "\n"), fontsize=7)
        ax.set_xlabel("Feature value", fontsize=7)
        if j == 0:
            ax.set_ylabel("SHAP value", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_shap_scatter_top5.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> {OUTPUT_DIR / '06_shap_scatter_top5.png'}")

    print("\nTop 10 most important features:")
    for i, (name, val) in enumerate(zip(top20_names[:10], top20_vals[:10]), 1):
        print(f"  {i:2d}. {name:<50s} mean|SHAP|={val:.4f}")

    print("\nFeature importance analysis complete. Figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
