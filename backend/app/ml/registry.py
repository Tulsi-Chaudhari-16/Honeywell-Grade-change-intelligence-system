"""
GCIS ML Pipeline — File-Based Model Registry

Lightweight registry that mimics MLflow's interface without requiring
a running MLflow server. In production this would be swapped for MLflow
with S3-backed artifact storage — the interface stays the same.

Registry layout on disk:
  models/registry/
    manifest.json              ← all model metadata
    lgbm_offspec/
      v1/
        model.pkl
        shap_explainer.pkl
        metrics.json
    tcn_forecast/
      v1/
        tcn_weights.pt
        lstm_weights.pt
        metrics.json
    isolation_forest/
      v1/
        model.pkl
    embedding_model/
      v1/
        encoder_weights.pt
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import MODEL_REGISTRY_PATH

MANIFEST_PATH = MODEL_REGISTRY_PATH / "manifest.json"


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MODEL_REGISTRY_PATH.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))


def save_model(
    name: str,
    version: str,
    artifact_path: Path,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """
    Register a model version. Copies artifact_path contents into the registry.

    Args:
        name: model name, e.g. 'lgbm_offspec'
        version: version string, e.g. 'v1' or '20240726_143200'
        artifact_path: directory containing model files to copy
        metrics: optional dict of evaluation metrics to store alongside

    Returns:
        Path to the stored model directory.
    """
    dest = MODEL_REGISTRY_PATH / name / version
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(artifact_path, dest)

    if metrics:
        (dest / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str)
        )

    manifest = _load_manifest()
    if name not in manifest:
        manifest[name] = {}
    manifest[name][version] = {
        "path": str(dest),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics or {},
    }
    # track which version is latest
    manifest[name]["_latest"] = version
    _save_manifest(manifest)
    print(f"[Registry] Saved {name} version={version} -> {dest}")
    return dest


def get_model_path(name: str, version: str = "latest") -> Path:
    """
    Retrieve the filesystem path for a registered model.

    Args:
        name: model name
        version: 'latest' (default) or specific version string

    Returns:
        Path to the model directory.

    Raises:
        FileNotFoundError if model or version not found.
    """
    manifest = _load_manifest()
    if name not in manifest:
        raise FileNotFoundError(f"No model named '{name}' in registry.")

    if version == "latest":
        version = manifest[name].get("_latest")
        if version is None:
            raise FileNotFoundError(f"No versions registered for model '{name}'.")

    if version not in manifest[name]:
        raise FileNotFoundError(
            f"Version '{version}' not found for model '{name}'."
        )

    path = Path(manifest[name][version]["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Registry entry exists but files missing at {path}. "
            "Re-run the training script to restore."
        )
    return path


def list_versions(name: str) -> list[dict]:
    """Return all registered versions for a model, newest first."""
    manifest = _load_manifest()
    if name not in manifest:
        return []
    return [
        {"version": v, **meta}
        for v, meta in manifest[name].items()
        if not v.startswith("_")
    ]


def get_metrics(name: str, version: str = "latest") -> dict:
    """Return the metrics dict stored for a given model version."""
    path = get_model_path(name, version)
    metrics_file = path / "metrics.json"
    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    return {}
