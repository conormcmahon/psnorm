"""Persist/load NormalizationModel as JSON, so a pipeline run can be resumed
without re-fitting scenes that already have a valid saved model."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .irmad import IrMadFit
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


def save_irmad_fit(fit: IrMadFit, path: str) -> None:
    """Persist a converged IrMadFit's small fitted parameters (rho, A, B,
    means, sigma — a handful of scalars and n_bands x n_bands matrices, not
    the pixel data itself) as JSON.

    This is what makes re-classifying a target's invariant pixels at a
    *different* `ncp_threshold` cheap after the fact (see
    normalize.reclassify_invariant_pixels): IR-MAD's iterative
    covariance/eigensolve refinement — the part that actually costs
    O(pixels * iterations) — never needs to re-run; only the fit's tiny
    parameters need to be reloaded, plus one fresh O(pixels) read+
    classification pass. All array fields are always host NumPy (see
    IrMadFit's docstring), so no device-specific handling is needed here
    regardless of what `device` the fit was originally computed on.
    """
    data = {
        "rho": fit.rho.tolist(),
        "A": fit.A.tolist(),
        "B": fit.B.tolist(),
        "means1": fit.means1.tolist(),
        "means2": fit.means2.tolist(),
        "sigma": fit.sigma.tolist(),
        "n_iterations": fit.n_iterations,
        "converged": fit.converged,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_irmad_fit(path: str) -> IrMadFit:
    with open(path) as f:
        data = json.load(f)
    return IrMadFit(
        rho=np.array(data["rho"]),
        A=np.array(data["A"]),
        B=np.array(data["B"]),
        means1=np.array(data["means1"]),
        means2=np.array(data["means2"]),
        sigma=np.array(data["sigma"]),
        n_iterations=data["n_iterations"],
        converged=data["converged"],
    )


def irmad_fit_is_valid(path: str) -> bool:
    """True if `path` exists and parses as a well-formed IrMadFit."""
    try:
        load_irmad_fit(path)
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
