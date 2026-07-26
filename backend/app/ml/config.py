"""
GCIS ML Pipeline — Centralised Configuration
All thresholds, paths, and tag names live here.
"""

from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[4]          # c:/Desktop/Honeywells
ML_DIR = Path(__file__).resolve().parent            # backend/app/ml/
MODEL_REGISTRY_PATH = ROOT / "models" / "registry"
SEED_DATA_PATH = ROOT / "data" / "seed"
NOTEBOOKS_PATH = ROOT / "data" / "notebooks"
ENVELOPE_PATH = ML_DIR / "envelopes"               # precomputed per-grade-pair envelopes

# ─── Sensor Tags ──────────────────────────────────────────────────────────────

ALL_TAGS = [
    "stock_flow",
    "filler_flow",
    "steam_pressure",
    "machine_speed",
    "moisture",
    "ash",
    "caliper",
    "dryer_steam_pressure",
    "reel_speed",
]

TARGET_TAG = "basis_weight"

# ─── Feature Engineering ──────────────────────────────────────────────────────

ROLLING_WINDOWS_SEC = [30, 120, 300]                # 30s, 2min, 5min
ROLLING_STATS = ["mean", "std", "min", "max", "rate_of_change"]

CROSS_RATIOS = [
    ("steam_pressure", "stock_flow"),               # steam:stock
    ("machine_speed", "steam_pressure"),            # speed:steam
    ("moisture", "steam_pressure"),                 # moisture:steam
]

TICK_INTERVAL_SEC = 5                               # sensor cadence

# ─── Model Thresholds ─────────────────────────────────────────────────────────

ALERT_THRESHOLD: float = 0.65                       # p_offspec >= this -> trigger RCA
MIN_CONFIDENCE: float = 0.40                        # below this -> LowConfidenceAlert only
OFFSPEC_BAND_PCT: float = 2.5                       # ±% of setpoint = off-spec

# ─── Drift Detection ──────────────────────────────────────────────────────────

PSI_RETRAIN_THRESHOLD: float = 0.20                 # PSI > 0.2 on any top-10 SHAP feature
NEW_EPISODES_RETRAIN_TRIGGER: int = 200             # OR 200 new confirmed episodes

# ─── TCN / Forecast ───────────────────────────────────────────────────────────

FORECAST_HORIZON_STEPS: int = 12                    # steps ahead (each = TICK_INTERVAL_SEC)
INPUT_WINDOW_STEPS: int = 60                        # 5min of history at 5s cadence
MC_DROPOUT_PASSES: int = 30                         # Monte Carlo passes for uncertainty

# ─── Embedding Model ──────────────────────────────────────────────────────────

EMBEDDING_DIM: int = 64                             # fixed-length transition embedding size

# ─── Isolation Forest ─────────────────────────────────────────────────────────

IF_CONTAMINATION: float = 0.05                      # expected fraction of anomalies

# ─── Synthetic Data ───────────────────────────────────────────────────────────

N_SYNTHETIC_TRANSITIONS: int = 500
MIN_TRANSITION_DURATION_MIN: float = 5.0
MAX_TRANSITION_DURATION_MIN: float = 30.0
OFFSPEC_RATE: float = 0.30                          # 30% of synthetic transitions are off-spec

GRADE_PAIRS = [
    ("newsprint_60", "newsprint_45"),
    ("newsprint_60", "directory_52"),
    ("directory_52", "super_calendered_56"),
    ("super_calendered_56", "newsprint_60"),
]

# ─── Operator Feedback ────────────────────────────────────────────────────────

OPERATOR_TIMEOUT_SEC: int = 180                     # 3 minutes before auto-dismiss
