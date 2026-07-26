"""
GCIS ML Pipeline — Synthetic Data Generator

Generates physically plausible paper mill grade-transition data.
No real mill data is available; this seed data pre-populates Postgres + Qdrant
for the demo. Clearly labeled as synthetic in all output files.

Physics model (simplified but plausible):
  Basis Weight ≈ (stock_flow + filler_flow) / machine_speed * K_bw
  Moisture     ≈ 100 * (1 - steam_efficiency(steam_pressure) * drying_factor)
  Ash          ≈ filler_flow / (stock_flow + filler_flow) * 100

A grade change is simulated as:
  1. Ramp phase    — setpoints ramp linearly from old -> new over ramp_duration
  2. Response phase— process variables follow with lag + noise
  3. Stabilize     — variables settle to new setpoints (or go off-spec)

Usage:
  python -m data.seed.generate_seed_data
  -> writes to data/seed/{sensor_readings.csv, transitions.csv,
                          training_examples.csv, qdrant_seed.json}
"""

from __future__ import annotations

import json
import random
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow running as a script from the repo root
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from backend.app.ml.config import (
    ALL_TAGS,
    GRADE_PAIRS,
    MAX_TRANSITION_DURATION_MIN,
    MIN_TRANSITION_DURATION_MIN,
    N_SYNTHETIC_TRANSITIONS,
    OFFSPEC_BAND_PCT,
    OFFSPEC_RATE,
    SEED_DATA_PATH,
    TICK_INTERVAL_SEC,
)

rng = np.random.default_rng(42)

# ─── Grade Definitions ────────────────────────────────────────────────────────
# Each grade has nominal setpoints for all process tags.
# Basis Weight target is the key quality metric (g/m²).

GRADE_SPECS: dict[str, dict[str, float]] = {
    "newsprint_60": {
        "basis_weight_target": 60.0,
        "stock_flow": 480.0,
        "filler_flow": 40.0,
        "steam_pressure": 3.2,
        "machine_speed": 1100.0,
        "moisture": 8.5,
        "ash": 7.5,
        "caliper": 82.0,
        "dryer_steam_pressure": 2.8,
        "reel_speed": 1095.0,
    },
    "newsprint_45": {
        "basis_weight_target": 45.0,
        "stock_flow": 360.0,
        "filler_flow": 28.0,
        "steam_pressure": 2.8,
        "machine_speed": 1200.0,
        "moisture": 8.0,
        "ash": 6.5,
        "caliper": 62.0,
        "dryer_steam_pressure": 2.4,
        "reel_speed": 1195.0,
    },
    "directory_52": {
        "basis_weight_target": 52.0,
        "stock_flow": 420.0,
        "filler_flow": 55.0,
        "steam_pressure": 3.0,
        "machine_speed": 1050.0,
        "moisture": 7.8,
        "ash": 11.0,
        "caliper": 72.0,
        "dryer_steam_pressure": 2.6,
        "reel_speed": 1045.0,
    },
    "super_calendered_56": {
        "basis_weight_target": 56.0,
        "stock_flow": 450.0,
        "filler_flow": 70.0,
        "steam_pressure": 3.5,
        "machine_speed": 980.0,
        "moisture": 6.5,
        "ash": 14.0,
        "caliper": 58.0,
        "dryer_steam_pressure": 3.1,
        "reel_speed": 975.0,
    },
}


# ─── Physics helpers ──────────────────────────────────────────────────────────


def _basis_weight(
    stock_flow: float,
    filler_flow: float,
    machine_speed: float,
    noise_std: float = 0.8,
) -> float:
    """Simplified BW equation. K_bw tuned to match grade specs approximately."""
    K_bw = 0.135
    bw = (stock_flow + filler_flow) / machine_speed * K_bw * 1000
    return bw + rng.normal(0, noise_std)


def _moisture(
    steam_pressure: float,
    machine_speed: float,
    noise_std: float = 0.15,
) -> float:
    steam_efficiency = min(1.0, steam_pressure / 4.0)
    drying = steam_efficiency * (1.0 - machine_speed / 2000.0) * 0.3
    m = 9.5 - drying * 10 + rng.normal(0, noise_std)
    return np.clip(m, 4.0, 14.0)


def _ash(
    filler_flow: float,
    stock_flow: float,
    noise_std: float = 0.2,
) -> float:
    total = stock_flow + filler_flow
    if total == 0:
        return 0.0
    ash = (filler_flow / total) * 100 + rng.normal(0, noise_std)
    return np.clip(ash, 0.0, 30.0)


# ─── Transition Simulation ────────────────────────────────────────────────────


@dataclass
class TransitionRecord:
    episode_id: str
    from_grade: str
    to_grade: str
    started_at: datetime
    ended_at: datetime
    outcome: str           # 'success' | 'offspec'
    final_basis_weight: float
    final_moisture: float
    final_ash: float
    recovery_time_min: float | None
    machine_id: str = "PM-01"


@dataclass
class SensorReading:
    episode_id: str
    ts: datetime
    tag_name: str
    value: float
    quality_flag: str = "good"
    is_imputed: bool = False


def simulate_transition(
    from_grade: str,
    to_grade: str,
    start_time: datetime,
    force_offspec: bool = False,
) -> tuple[TransitionRecord, list[SensorReading]]:
    """
    Simulate a single grade transition.

    Returns:
        (TransitionRecord, list of SensorReading rows)
    """
    ep_id = str(uuid.uuid4())
    old_sp = GRADE_SPECS[from_grade]
    new_sp = GRADE_SPECS[to_grade]

    duration_min = rng.uniform(MIN_TRANSITION_DURATION_MIN, MAX_TRANSITION_DURATION_MIN)
    n_ticks = int(duration_min * 60 / TICK_INTERVAL_SEC)

    # Ramp duration: 30–60% of total
    ramp_ticks = int(n_ticks * rng.uniform(0.30, 0.60))

    # Off-spec transitions get additional disturbances
    if force_offspec:
        disturb_tag = rng.choice(ALL_TAGS[:4])  # disturb a flow/pressure tag
        disturb_magnitude = rng.uniform(0.10, 0.25) * old_sp[disturb_tag]
        disturb_direction = rng.choice([-1, 1])
    else:
        disturb_tag = None
        disturb_magnitude = 0.0
        disturb_direction = 0

    readings: list[SensorReading] = []
    bw_target_new = new_sp["basis_weight_target"]
    bw_final = None
    moisture_final = None
    ash_final = None
    recovery_tick: int | None = None

    # Process variables lag behind setpoints
    lag_factor = rng.uniform(0.04, 0.12)  # exponential smoothing α

    current = {tag: old_sp[tag] for tag in ALL_TAGS}

    for tick in range(n_ticks):
        ts = start_time + timedelta(seconds=tick * TICK_INTERVAL_SEC)
        progress = tick / max(ramp_ticks, 1)

        # Target setpoints at this tick (linear ramp then hold)
        if tick < ramp_ticks:
            target = {
                tag: old_sp[tag] + (new_sp[tag] - old_sp[tag]) * progress
                for tag in ALL_TAGS
            }
        else:
            target = {tag: new_sp[tag] for tag in ALL_TAGS}

        # Apply disturbance in off-spec transitions (during first half of ramp)
        if force_offspec and disturb_tag and tick < ramp_ticks // 2:
            target[disturb_tag] += disturb_direction * disturb_magnitude * (
                1 - progress * 2
            )

        # Simulate process lag (exponential smoothing)
        for tag in ALL_TAGS:
            noise = rng.normal(0, 0.005 * abs(target[tag]) + 0.01)
            current[tag] = (
                current[tag] * (1 - lag_factor) + target[tag] * lag_factor + noise
            )

        # Simulate occasional sensor dropout (~1% of ticks, random tag)
        dropout_tag: str | None = None
        if rng.random() < 0.01:
            dropout_tag = rng.choice(ALL_TAGS)

        for tag in ALL_TAGS:
            if tag == dropout_tag:
                # Stale / missing reading
                readings.append(
                    SensorReading(
                        episode_id=ep_id,
                        ts=ts,
                        tag_name=tag,
                        value=current[tag],
                        quality_flag="stale",
                        is_imputed=True,
                    )
                )
            else:
                readings.append(
                    SensorReading(
                        episode_id=ep_id,
                        ts=ts,
                        tag_name=tag,
                        value=round(current[tag], 4),
                        quality_flag="good",
                    )
                )

        # Compute derived quality metrics at this tick
        bw = _basis_weight(
            current["stock_flow"],
            current["filler_flow"],
            current["machine_speed"],
        )
        bw_deviation_pct = abs(bw - bw_target_new) / bw_target_new * 100
        bw_final = bw
        moisture_final = _moisture(current["steam_pressure"], current["machine_speed"])
        ash_final = _ash(current["filler_flow"], current["stock_flow"])

        # Track when BW first stabilises within band (for success transitions)
        if not force_offspec and bw_deviation_pct <= OFFSPEC_BAND_PCT and tick >= ramp_ticks:
            if recovery_tick is None:
                recovery_tick = tick

        # Add BW as a tag reading (it's the output variable)
        readings.append(
            SensorReading(
                episode_id=ep_id,
                ts=ts,
                tag_name="basis_weight",
                value=round(bw, 4),
                quality_flag="good",
            )
        )

    end_time = start_time + timedelta(seconds=n_ticks * TICK_INTERVAL_SEC)

    # Determine outcome
    bw_final_dev = abs(bw_final - bw_target_new) / bw_target_new * 100
    if force_offspec or bw_final_dev > OFFSPEC_BAND_PCT:
        outcome = "offspec"
        recovery_time = None
    else:
        outcome = "success"
        recovery_time = (
            (recovery_tick * TICK_INTERVAL_SEC / 60) if recovery_tick else duration_min
        )

    record = TransitionRecord(
        episode_id=ep_id,
        from_grade=from_grade,
        to_grade=to_grade,
        started_at=start_time,
        ended_at=end_time,
        outcome=outcome,
        final_basis_weight=round(bw_final, 3),
        final_moisture=round(moisture_final, 3),
        final_ash=round(ash_final, 3),
        recovery_time_min=round(recovery_time, 2) if recovery_time else None,
    )
    return record, readings


# ─── Label generation ─────────────────────────────────────────────────────────


def _make_training_labels(
    readings: list[SensorReading],
    record: TransitionRecord,
    lookahead_sec: int = 300,  # 5 minutes ahead
) -> list[dict]:
    """
    For each tick, label whether the transition goes off-spec within
    the next `lookahead_sec` seconds. This is the binary classification label.
    """
    bw_readings = {r.ts: r.value for r in readings if r.tag_name == "basis_weight"}
    timestamps = sorted(bw_readings.keys())
    target_bw = GRADE_SPECS[record.to_grade]["basis_weight_target"]

    labels = []
    for ts in timestamps:
        cutoff = ts + timedelta(seconds=lookahead_sec)
        future_bws = [bw_readings[t] for t in timestamps if ts < t <= cutoff]
        if not future_bws:
            continue

        offspec = any(
            abs(bw - target_bw) / target_bw * 100 > OFFSPEC_BAND_PCT
            for bw in future_bws
        )
        labels.append(
            {
                "example_id": str(uuid.uuid4()),
                "episode_id": record.episode_id,
                "ts": ts.isoformat(),
                "label_offspec": offspec,
                "label_source": "predicted",  # synthetic = no real lab confirmation
            }
        )
    return labels


# ─── Main generation loop ─────────────────────────────────────────────────────


def generate_all(n: int = N_SYNTHETIC_TRANSITIONS) -> None:
    """Generate N synthetic transitions and write CSV + JSON seed files."""
    SEED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    n_offspec = int(n * OFFSPEC_RATE)
    n_success = n - n_offspec

    all_transitions: list[dict] = []
    all_readings: list[dict] = []
    all_labels: list[dict] = []
    qdrant_seed: list[dict] = []

    # Start historical timestamps ~1 year back
    now = datetime.now(timezone.utc)
    current_time = now - timedelta(days=365)

    print(f"Generating {n} synthetic transitions ({n_offspec} off-spec, {n_success} success)...")

    for i in range(n):
        from_grade, to_grade = random.choice(GRADE_PAIRS)
        force_offspec = i < n_offspec  # first n_offspec are forced off-spec, rest random

        record, readings = simulate_transition(
            from_grade=from_grade,
            to_grade=to_grade,
            start_time=current_time,
            force_offspec=force_offspec,
        )

        labels = _make_training_labels(readings, record)

        all_transitions.append(
            {
                "episode_id": record.episode_id,
                "machine_id": record.machine_id,
                "from_grade": record.from_grade,
                "to_grade": record.to_grade,
                "started_at": record.started_at.isoformat(),
                "ended_at": record.ended_at.isoformat(),
                "outcome": record.outcome,
                "recovery_time_min": record.recovery_time_min,
                "final_basis_weight": record.final_basis_weight,
                "final_moisture": record.final_moisture,
                "final_ash": record.final_ash,
                "embedding_id": None,  # filled after embedding model runs
                "synthetic": True,
            }
        )
        all_readings.extend(
            {
                "episode_id": r.episode_id,
                "ts": r.ts.isoformat(),
                "tag_name": r.tag_name,
                "value": r.value,
                "quality_flag": r.quality_flag,
                "is_imputed": r.is_imputed,
            }
            for r in readings
        )
        all_labels.extend(labels)

        # Qdrant seed entry (embedding placeholder — real vector filled by train_embedding.py)
        qdrant_seed.append(
            {
                "episode_id": record.episode_id,
                "from_grade": from_grade,
                "to_grade": to_grade,
                "outcome": record.outcome,
                "recovery_time_min": record.recovery_time_min,
                "final_basis_weight": record.final_basis_weight,
                "embedding_vector": None,   # placeholder
            }
        )

        # Advance time (gap between transitions: 15min–4hrs)
        gap_min = random.uniform(15, 240)
        current_time = record.ended_at + timedelta(minutes=gap_min)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} transitions generated...")

    # ── Write outputs ──────────────────────────────────────────────────────────
    pd.DataFrame(all_transitions).to_csv(
        SEED_DATA_PATH / "transitions.csv", index=False
    )
    print(f"Wrote transitions.csv ({len(all_transitions)} rows)")

    pd.DataFrame(all_readings).to_csv(
        SEED_DATA_PATH / "sensor_readings.csv", index=False
    )
    print(f"Wrote sensor_readings.csv ({len(all_readings)} rows)")

    pd.DataFrame(all_labels).to_csv(
        SEED_DATA_PATH / "training_examples.csv", index=False
    )
    print(f"Wrote training_examples.csv ({len(all_labels)} rows)")

    (SEED_DATA_PATH / "qdrant_seed.json").write_text(
        json.dumps(qdrant_seed, indent=2, default=str)
    )
    print(f"Wrote qdrant_seed.json ({len(qdrant_seed)} entries)")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "synthetic": True,
        "generated_at": now.isoformat(),
        "n_transitions": n,
        "n_offspec": sum(1 for t in all_transitions if t["outcome"] == "offspec"),
        "n_success": sum(1 for t in all_transitions if t["outcome"] == "success"),
        "grade_pairs": GRADE_PAIRS,
        "n_sensor_readings": len(all_readings),
        "n_training_labels": len(all_labels),
        "tick_interval_sec": TICK_INTERVAL_SEC,
        "note": (
            "This dataset is SYNTHETIC. Generated from physically plausible "
            "process equations, not real mill telemetry. Clearly labeled as such "
            "in all downstream artefacts."
        ),
    }
    (SEED_DATA_PATH / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Metadata -> {SEED_DATA_PATH / 'metadata.json'}")


if __name__ == "__main__":
    generate_all()
