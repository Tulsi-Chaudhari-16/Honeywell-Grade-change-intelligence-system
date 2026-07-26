"""
GCIS ML Pipeline — Feature Engineering

Computes the full feature vector from a rolling buffer of sensor readings
for a single active transition episode.

Key feature groups:
  1. Rolling statistics  (30s / 2min / 5min windows)
  2. Cross-variable ratios
  3. Recipe-delta features
  4. Historical-envelope distance (precomputed per grade pair)

Called by the Feature Engineering Agent at each tick.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .config import (
    ALL_TAGS,
    CROSS_RATIOS,
    ENVELOPE_PATH,
    ROLLING_STATS,
    ROLLING_WINDOWS_SEC,
    TICK_INTERVAL_SEC,
)
from .types import FeatureVector


# ─── Envelope cache (loaded once per grade pair) ──────────────────────────────


_envelope_cache: dict[str, np.ndarray] = {}


def _load_envelope(grade_pair: str) -> np.ndarray | None:
    """
    Load precomputed mean successful-transition trajectory for a grade pair.
    Shape: (n_steps, n_tags) — mean values per tag at each step.
    """
    if grade_pair in _envelope_cache:
        return _envelope_cache[grade_pair]

    path = ENVELOPE_PATH / f"{grade_pair}.npy"
    if path.exists():
        env = np.load(path)
        _envelope_cache[grade_pair] = env
        return env
    return None  # envelope not yet computed — first run


# ─── Rolling Buffer ───────────────────────────────────────────────────────────


class TagBuffer:
    """
    Maintains a fixed-length rolling buffer of (timestamp, value) pairs
    for a single sensor tag. Thread-unsafe — one buffer per agent thread.
    """

    def __init__(self, max_seconds: int = 360) -> None:
        # max_seconds = largest window + margin
        self._max_len = max_seconds // TICK_INTERVAL_SEC + 1
        self._buf: deque[tuple[float, float]] = deque(maxlen=self._max_len)

    def push(self, ts: float, value: float) -> None:
        self._buf.append((ts, value))

    def window(self, seconds: int) -> np.ndarray:
        """Return values within the last `seconds` seconds."""
        if not self._buf:
            return np.array([])
        cutoff = self._buf[-1][0] - seconds
        return np.array([v for t, v in self._buf if t >= cutoff])

    @property
    def last(self) -> float | None:
        return self._buf[-1][1] if self._buf else None


class EpisodeBuffers:
    """One TagBuffer per sensor tag for a single episode."""

    def __init__(self) -> None:
        self.bufs: dict[str, TagBuffer] = {tag: TagBuffer() for tag in ALL_TAGS}
        self.start_ts: float | None = None

    def push(self, ts: float, tag: str, value: float) -> None:
        if self.start_ts is None:
            self.start_ts = ts
        if tag in self.bufs:
            self.bufs[tag].push(ts, value)

    def time_since_start(self, ts: float) -> float:
        if self.start_ts is None:
            return 0.0
        return ts - self.start_ts


# ─── Core feature computation ─────────────────────────────────────────────────


def _rolling_stats(values: np.ndarray) -> dict[str, float]:
    """Compute rolling stats for a window of values. Returns NaN for empty windows."""
    if len(values) == 0:
        return {stat: float("nan") for stat in ROLLING_STATS}

    out: dict[str, float] = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "rate_of_change": float(values[-1] - values[0]) if len(values) > 1 else 0.0,
    }
    return out


def compute_features(
    buffers: EpisodeBuffers,
    ts: float,
    old_recipe: dict[str, float],
    new_recipe: dict[str, float],
    grade_pair: str,
) -> FeatureVector:
    """
    Compute the full feature vector for the current tick.

    Args:
        buffers: Rolling tag buffers for the episode.
        ts: Current Unix timestamp.
        old_recipe: Setpoint values for the source grade.
        new_recipe: Setpoint values for the target grade.
        grade_pair: e.g. 'newsprint_60__directory_52' — used for envelope lookup.

    Returns:
        FeatureVector with all features and imputation flags.
    """
    features: dict[str, float] = {}
    imputation_flags: dict[str, bool] = {}

    # ── 1. Rolling statistics ──────────────────────────────────────────────────
    for tag in ALL_TAGS:
        buf = buffers.bufs[tag]
        last_val = buf.last

        for window_sec in ROLLING_WINDOWS_SEC:
            window_vals = buf.window(window_sec)
            imputed = len(window_vals) == 0

            if imputed and last_val is not None:
                # hold last known value
                window_vals = np.array([last_val])
            elif imputed:
                window_vals = np.array([0.0])

            stats = _rolling_stats(window_vals)
            for stat, val in stats.items():
                key = f"{tag}_{stat}_{window_sec}s"
                features[key] = val if not np.isnan(val) else 0.0
                if imputed:
                    imputation_flags[key] = True

    # ── 2. Cross-variable ratios ───────────────────────────────────────────────
    for tag_a, tag_b in CROSS_RATIOS:
        val_a = buffers.bufs[tag_a].last
        val_b = buffers.bufs[tag_b].last
        key = f"ratio_{tag_a}__{tag_b}"
        if val_a is not None and val_b is not None and val_b != 0:
            features[key] = val_a / val_b
        else:
            features[key] = 0.0
            imputation_flags[key] = True

    # ── 3. Recipe-delta features ───────────────────────────────────────────────
    for tag in ALL_TAGS:
        current = buffers.bufs[tag].last
        if current is None:
            current = 0.0
            imputation_flags[f"delta_from_old_{tag}"] = True
            imputation_flags[f"delta_from_new_{tag}"] = True

        old_sp = old_recipe.get(tag, current)
        new_sp = new_recipe.get(tag, current)
        features[f"delta_from_old_{tag}"] = current - old_sp
        features[f"delta_from_new_{tag}"] = current - new_sp

    features["time_since_transition_start_sec"] = buffers.time_since_start(ts)

    # ── 4. Historical-envelope distance ───────────────────────────────────────
    envelope = _load_envelope(grade_pair)
    if envelope is not None:
        step_idx = min(
            int(buffers.time_since_start(ts) / TICK_INTERVAL_SEC),
            len(envelope) - 1,
        )
        current_vals = np.array(
            [buffers.bufs[tag].last or 0.0 for tag in ALL_TAGS]
        )
        envelope_mean = envelope[step_idx]
        dist = float(np.linalg.norm(current_vals - envelope_mean))
        features["envelope_distance"] = dist
    else:
        features["envelope_distance"] = 0.0

    return FeatureVector(
        episode_id="",           # filled in by the Feature Engineering Agent
        ts=datetime.utcfromtimestamp(ts),
        features=features,
        imputation_flags=imputation_flags,
    )


# ─── Envelope pre-computation (run offline once per grade pair) ───────────────


def compute_and_save_envelope(
    transitions_df: pd.DataFrame,
    grade_pair: str,
    max_steps: int = 360,
) -> None:
    """
    Compute the mean successful-transition trajectory for a grade pair
    and save it to the ENVELOPE_PATH for use during live inference.

    Args:
        transitions_df: DataFrame with columns [episode_id, step, <tag>...]
                        for successful transitions only.
        grade_pair: identifier used as the filename.
        max_steps: pad/truncate trajectories to this length.
    """
    ENVELOPE_PATH.mkdir(parents=True, exist_ok=True)

    episodes = transitions_df["episode_id"].unique()
    all_trajectories: list[np.ndarray] = []

    for ep_id in episodes:
        ep = transitions_df[transitions_df["episode_id"] == ep_id].sort_values("step")
        vals = ep[ALL_TAGS].values  # shape (n_steps, n_tags)
        # Pad or truncate to max_steps
        if len(vals) >= max_steps:
            vals = vals[:max_steps]
        else:
            pad = np.tile(vals[-1], (max_steps - len(vals), 1))
            vals = np.vstack([vals, pad])
        all_trajectories.append(vals)

    if not all_trajectories:
        print(f"[Envelope] No successful transitions for {grade_pair}, skipping.")
        return

    envelope = np.mean(all_trajectories, axis=0)  # (max_steps, n_tags)
    out_path = ENVELOPE_PATH / f"{grade_pair}.npy"
    np.save(out_path, envelope)
    print(f"[Envelope] Saved envelope for {grade_pair} -> {out_path}")


# ─── Batch feature extraction (for training) ─────────────────────────────────


def extract_features_from_df(
    readings_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    recipes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Batch feature extraction over historical sensor readings for model training.
    Vectorized for performance over millions of ticks.
    """
    import time
    
    print("[Extract] Pivoting readings...")
    wide = readings_df.pivot_table(
        index=["episode_id", "ts"],
        columns="tag_name",
        values="value"
    ).reset_index().sort_values(["episode_id", "ts"])

    print("[Extract] Joining metadata...")
    trans = transitions_df[["episode_id", "started_at", "from_grade", "to_grade"]]
    wide = wide.merge(trans, on="episode_id")
    wide["ts"] = pd.to_datetime(wide["ts"])
    wide["started_at"] = pd.to_datetime(wide["started_at"])
    wide["time_since_transition_start_sec"] = (wide["ts"] - wide["started_at"]).dt.total_seconds()
    
    wide = wide.sort_values(["episode_id", "ts"])
    groups = wide.groupby("episode_id")

    # Impute missing intra-episode values (forward fill, then fillna 0 for start of episode)
    for tag in ALL_TAGS:
        wide[tag] = groups[tag].ffill().fillna(0.0)

    print("[Extract] Computing rolling features...")
    # ── 1. Rolling statistics ──────────────────────────────────────────────────
    for window_sec in ROLLING_WINDOWS_SEC:
        window_ticks = window_sec // TICK_INTERVAL_SEC
        r = groups[ALL_TAGS].rolling(window=window_ticks, min_periods=1)
        
        means = r.mean().reset_index(level=0, drop=True)
        stds = r.std().reset_index(level=0, drop=True).fillna(0.0)
        mins = r.min().reset_index(level=0, drop=True)
        maxs = r.max().reset_index(level=0, drop=True)
        
        first_in_window = groups[ALL_TAGS].shift(window_ticks - 1)
        first_in_group = groups[ALL_TAGS].transform('first')
        first_in_window = first_in_window.fillna(first_in_group)
        
        rate_of_change = wide[ALL_TAGS] - first_in_window
        
        for tag in ALL_TAGS:
            wide[f"{tag}_mean_{window_sec}s"] = means[tag]
            wide[f"{tag}_std_{window_sec}s"] = stds[tag]
            wide[f"{tag}_min_{window_sec}s"] = mins[tag]
            wide[f"{tag}_max_{window_sec}s"] = maxs[tag]
            wide[f"{tag}_rate_of_change_{window_sec}s"] = rate_of_change[tag]

    print("[Extract] Computing cross-variable ratios & recipe deltas...")
    # ── 2. Cross-variable ratios ───────────────────────────────────────────────
    for tag_a, tag_b in CROSS_RATIOS:
        key = f"ratio_{tag_a}__{tag_b}"
        wide[key] = np.where(wide[tag_b] != 0, wide[tag_a] / wide[tag_b], 0.0)

    # ── 3. Recipe-delta features ───────────────────────────────────────────────
    recipes = recipes_df.set_index("grade_code")
    for tag in ALL_TAGS:
        old_sp = wide["from_grade"].map(recipes[tag]).fillna(wide[tag])
        new_sp = wide["to_grade"].map(recipes[tag]).fillna(wide[tag])
        
        wide[f"delta_from_old_{tag}"] = wide[tag] - old_sp
        wide[f"delta_from_new_{tag}"] = wide[tag] - new_sp

    print("[Extract] Computing envelope distance...")
    # ── 4. Historical-envelope distance ───────────────────────────────────────
    wide["envelope_distance"] = 0.0
    for grade_pair in wide[["from_grade", "to_grade"]].drop_duplicates().itertuples(index=False):
        gp_str = f"{grade_pair.from_grade}__{grade_pair.to_grade}"
        envelope = _load_envelope(gp_str)
        if envelope is not None:
            mask = (wide["from_grade"] == grade_pair.from_grade) & (wide["to_grade"] == grade_pair.to_grade)
            gp_data = wide[mask]
            
            step_idx = (gp_data["time_since_transition_start_sec"] / TICK_INTERVAL_SEC).astype(int)
            step_idx = np.clip(step_idx, 0, len(envelope) - 1)
            
            env_means = envelope[step_idx]
            curr_vals = gp_data[ALL_TAGS].values
            dists = np.linalg.norm(curr_vals - env_means, axis=1)
            wide.loc[mask, "envelope_distance"] = dists

    # ── Imputation Flags ───────────────────────────────────────────────────────
    is_imp = readings_df.groupby(["episode_id", "ts"])["is_imputed"].any().reset_index()
    is_imp["ts"] = pd.to_datetime(is_imp["ts"])
    wide = wide.merge(is_imp, on=["episode_id", "ts"], how="left")
    wide["any_imputed"] = wide["is_imputed"].fillna(False)

    drop_cols = ["started_at", "from_grade", "to_grade", "is_imputed", "basis_weight"] + ALL_TAGS
    wide = wide.drop(columns=drop_cols, errors="ignore")
    
    return wide
