"""
GCIS — EDA: Synthetic Seed Data

Explores the generated synthetic dataset:
  - Transition outcome distribution
  - Basis Weight trajectories by outcome
  - Per-tag distribution plots
  - Grade pair breakdown

Run: python data/notebooks/01_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from backend.app.ml.config import ALL_TAGS, SEED_DATA_PATH

plt.style.use("dark_background")
sns.set_palette("husl")

OUTPUT_DIR = SEED_DATA_PATH.parent / "notebooks" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
transitions = pd.read_csv(SEED_DATA_PATH / "transitions.csv", parse_dates=["started_at", "ended_at"])
readings = pd.read_csv(SEED_DATA_PATH / "sensor_readings.csv", parse_dates=["ts"])
labels = pd.read_csv(SEED_DATA_PATH / "training_examples.csv")

print(f"Transitions: {len(transitions)} | Readings: {len(readings)} | Labels: {len(labels)}")
print("\nOutcome distribution:")
print(transitions["outcome"].value_counts())

# ── 1. Outcome distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("GCIS Synthetic Dataset — EDA Overview", fontsize=14, y=1.02)

outcome_counts = transitions["outcome"].value_counts()
axes[0].pie(outcome_counts, labels=outcome_counts.index, autopct="%1.1f%%", startangle=90)
axes[0].set_title("Transition Outcomes")

# ── 2. Duration distribution by outcome ───────────────────────────────────────
transitions["duration_min"] = (
    transitions["ended_at"] - transitions["started_at"]
).dt.total_seconds() / 60

for outcome, grp in transitions.groupby("outcome"):
    axes[1].hist(grp["duration_min"], bins=20, alpha=0.7, label=outcome)
axes[1].set_xlabel("Duration (min)")
axes[1].set_title("Transition Duration by Outcome")
axes[1].legend()

# ── 3. Grade pair distribution ────────────────────────────────────────────────
transitions["grade_pair"] = transitions["from_grade"] + " -> " + transitions["to_grade"]
pair_counts = transitions["grade_pair"].value_counts()
axes[2].barh(pair_counts.index, pair_counts.values)
axes[2].set_title("Grade Pair Distribution")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_overview.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_DIR / '01_overview.png'}")

# ── 4. BW trajectories: success vs offspec ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
bw_readings = readings[readings["tag_name"] == "basis_weight"].copy()
bw_readings = bw_readings.merge(transitions[["episode_id", "outcome"]], on="episode_id")

# Sample 10 transitions per outcome for clarity
for outcome, color in [("success", "#22d3ee"), ("offspec", "#f97316")]:
    sample_eps = (
        transitions[transitions["outcome"] == outcome]["episode_id"].sample(min(10, len(transitions[transitions["outcome"] == outcome])), random_state=42).tolist()
    )
    for ep_id in sample_eps:
        ep_bw = bw_readings[bw_readings["episode_id"] == ep_id].sort_values("ts")
        ax.plot(range(len(ep_bw)), ep_bw["value"], color=color, alpha=0.4, linewidth=0.8)

ax.axhline(y=0, color="white", linestyle="--", alpha=0.3)
ax.set_xlabel("Tick")
ax.set_ylabel("Basis Weight (g/m²)")
ax.set_title("BW Trajectories — Success (cyan) vs Off-spec (orange)")
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#22d3ee", label="Success"),
    Line2D([0], [0], color="#f97316", label="Off-spec"),
]
ax.legend(handles=legend_elements)
plt.savefig(OUTPUT_DIR / "02_bw_trajectories.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_DIR / '02_bw_trajectories.png'}")

# ── 5. Per-tag distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle("Per-Tag Value Distributions", fontsize=13)
tag_readings = readings[readings["tag_name"].isin(ALL_TAGS)]
tag_readings = tag_readings.merge(transitions[["episode_id", "outcome"]], on="episode_id")

for i, (tag, ax) in enumerate(zip(ALL_TAGS, axes.flatten())):
    tag_data = tag_readings[tag_readings["tag_name"] == tag]
    for outcome, color in [("success", "#22d3ee"), ("offspec", "#f97316")]:
        grp = tag_data[tag_data["outcome"] == outcome]["value"]
        ax.hist(grp, bins=40, alpha=0.6, color=color, label=outcome, density=True)
    ax.set_title(tag.replace("_", " ").title())
    ax.set_xlabel("Value")
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_tag_distributions.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_DIR / '03_tag_distributions.png'}")

# ── 6. Label balance ──────────────────────────────────────────────────────────
offspec_rate = labels["label_offspec"].mean()
print(f"\nLabel balance: {offspec_rate:.2%} off-spec ticks")
print(f"Total training examples: {len(labels)}")

print("\nEDA complete. Figures saved to:", OUTPUT_DIR)
