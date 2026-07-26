"""
GCIS ML Pipeline — Train Transition Embedding Model + Generate Qdrant Seed Vectors

End-to-end:
  1. Load per-episode feature sequences from seed data
  2. Train 1D-CNN autoencoder
  3. Encode all historical transitions -> fixed-length vectors
  4. Write vectors back to qdrant_seed.json for Qdrant indexing

Run from repo root:
    python -m backend.app.ml.train.train_embedding
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.app.ml.config import ALL_TAGS, INPUT_WINDOW_STEPS, SEED_DATA_PATH
from backend.app.ml.features import extract_features_from_df
from backend.app.ml.models.embedding_model import TransitionEmbedder, train as embed_train
from backend.app.ml.registry import save_model, get_model_path


def _build_episode_sequences(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = INPUT_WINDOW_STEPS,
) -> tuple[np.ndarray, list[str]]:
    """
    Take the last `seq_len` ticks of each episode and stack them.

    Returns:
        X: (n_episodes, n_features, seq_len)
        episode_ids: matching list of episode IDs
    """
    X_list, ep_ids = [], []
    for ep_id, ep_df in features_df.groupby("episode_id"):
        ep_df = ep_df.sort_values("ts")
        vals = ep_df[feature_cols].fillna(0.0).values  # (n_ticks, n_features)
        if len(vals) < seq_len:
            # Pad from the left with first value
            pad = np.tile(vals[0], (seq_len - len(vals), 1))
            vals = np.vstack([pad, vals])
        else:
            vals = vals[-seq_len:]
        X_list.append(vals.T)   # (n_features, seq_len)
        ep_ids.append(ep_id)
    return np.array(X_list, dtype=np.float32), ep_ids


def main() -> None:
    print("=" * 60)
    print("GCIS — Training Transition Embedding Model")
    print("=" * 60)

    readings = pd.read_csv(SEED_DATA_PATH / "sensor_readings.csv", parse_dates=["ts"])
    transitions = pd.read_csv(SEED_DATA_PATH / "transitions.csv", parse_dates=["started_at", "ended_at"])

    from data.seed.generate_seed_data import GRADE_SPECS
    grade_codes = list(GRADE_SPECS.keys())
    recipes = pd.DataFrame(
        [{"grade_code": g, **{tag: GRADE_SPECS[g].get(tag, 0.0) for tag in ALL_TAGS}}
         for g in grade_codes]
    )

    print(f"Loaded: {len(transitions)} transitions")

    # ── Extract features ───────────────────────────────────────────────────────
    print("\nExtracting features...")
    features_df = extract_features_from_df(readings, transitions, recipes)
    feature_cols = [
        c for c in features_df.columns
        if c not in ("episode_id", "ts", "any_imputed")
    ]
    features_df["ts"] = pd.to_datetime(features_df["ts"])

    # Normalise
    scaler = StandardScaler()
    features_df[feature_cols] = scaler.fit_transform(
        features_df[feature_cols].fillna(0.0)
    )

    # ── Build per-episode sequences ────────────────────────────────────────────
    print("Building episode sequences...")
    X, episode_ids = _build_episode_sequences(features_df, feature_cols)
    print(f"Sequences shape: {X.shape}")  # (n_episodes, n_features, seq_len)

    # ── Train / Val split ──────────────────────────────────────────────────────
    X_train, X_val, ids_train, ids_val = train_test_split(
        X, episode_ids, test_size=0.2, random_state=42
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        import pickle
        with open(tmp_path / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        (tmp_path / "feature_columns.txt").write_text("\n".join(feature_cols))

        metrics = embed_train(X_train, X_val, tmp_path)
        save_model("embedding_model", "v1", tmp_path, metrics)

    print(f"\n[Success] Embedding model trained. Best val loss: {metrics['best_val_reconstruction_loss']}")

    # ── Generate embeddings for ALL historical transitions -> Qdrant seed ──────
    print("\nGenerating embeddings for all transitions...")
    model_dir = get_model_path("embedding_model", "v1")
    embedder = TransitionEmbedder.load(model_dir)

    embeddings: dict[str, list[float]] = {}
    for i, (seq, ep_id) in enumerate(zip(X, episode_ids)):
        vec = embedder.embed(seq)
        embeddings[ep_id] = vec.tolist()
        if (i + 1) % 100 == 0:
            print(f"  Embedded {i + 1}/{len(episode_ids)} transitions...")

    # ── Update qdrant_seed.json with real vectors ──────────────────────────────
    qdrant_seed_path = SEED_DATA_PATH / "qdrant_seed.json"
    qdrant_seed = json.loads(qdrant_seed_path.read_text())
    for entry in qdrant_seed:
        ep_id = entry["episode_id"]
        if ep_id in embeddings:
            entry["embedding_vector"] = embeddings[ep_id]

    qdrant_seed_path.write_text(json.dumps(qdrant_seed, indent=2))
    print(f"[Success] Updated qdrant_seed.json with {len(embeddings)} embedding vectors.")
    print("  -> Ready to index into Qdrant (partner's Qdrant seeding script reads this file).")


if __name__ == "__main__":
    main()
