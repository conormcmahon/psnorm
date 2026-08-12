"""Persist/load NormalizationModel as JSON, so a pipeline run can be resumed
without re-fitting scenes that already have a valid saved model."""

from __future__ import annotations

import json
from dataclasses import asdict

from .normalize import BandModel, NormalizationModel


def save_model(model: NormalizationModel, path: str) -> None:
    with open(path, "w") as f:
        json.dump(asdict(model), f, indent=2)


def load_model(path: str) -> NormalizationModel:
    with open(path) as f:
        data = json.load(f)
    bands = [BandModel(**b) for b in data.pop("bands")]
    return NormalizationModel(bands=bands, **data)


def model_is_valid(path: str) -> bool:
    """True if `path` exists and parses as a well-formed NormalizationModel."""
    try:
        load_model(path)
        return True
    except Exception:
        return False


def save_json(data: dict, path: str) -> None:
    """Generic small-record persistence, used for Phase A candidate-detection
    stats (IR-MAD diagnostics) that aren't a NormalizationModel."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
