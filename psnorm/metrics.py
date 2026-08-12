"""Per-band and pooled R^2 (Pearson correlation strength) and RMSE (direct
DN-difference agreement) between two rasters over a shared valid-pixel mask.

Deliberately takes an already-computed overlap window + validity mask rather
than computing them itself: the same window/mask (built once from the *raw*
scenes' UDM2 + OmniCloudMask) is reused for both the "before" (raw vs raw)
and "after" (normalized vs normalized) comparison of a scene pair, since
apply.apply_model() preserves georeferencing exactly and a linear DN rescale
doesn't change which pixels are cloudy. That avoids recomputing the (fairly
expensive) OmniCloudMask pass twice per pair and keeps the comparison
apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import io, sensors


@dataclass
class BandAgreement:
    band_name: str
    r2: float
    rmse: float


@dataclass
class AgreementResult:
    n_pixels: int
    bands: list[BandAgreement]
    pooled_r2: float
    pooled_rmse: float


def _r2_rmse(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.size < 2:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    if np.std(a) == 0 or np.std(b) == 0:
        r2 = float("nan")
    else:
        r = np.corrcoef(a, b)[0, 1]
        r2 = float(r * r)
    return r2, rmse


def agreement_between(
    path_a: str,
    window_a: io.Window,
    band_names_a: list[str],
    path_b: str,
    window_b: io.Window,
    band_names_b: list[str],
    valid_mask: np.ndarray,
    *,
    common_bands: list[str] | None = None,
) -> AgreementResult:
    if common_bands is None:
        common_bands = [b for b in band_names_a if b in band_names_b]
    if not common_bands:
        raise ValueError(
            f"No shared band names between {band_names_a!r} and {band_names_b!r}."
        )

    n_pixels = int(valid_mask.sum())
    band_results = []
    pooled_a, pooled_b = [], []
    for name in common_bands:
        idx_a = sensors.band_index(band_names_a, name)
        idx_b = sensors.band_index(band_names_b, name)
        arr_a = io.read_window_bands(path_a, window_a, [idx_a])[0][valid_mask]
        arr_b = io.read_window_bands(path_b, window_b, [idx_b])[0][valid_mask]
        r2, rmse = _r2_rmse(arr_a, arr_b)
        band_results.append(BandAgreement(band_name=name, r2=r2, rmse=rmse))
        pooled_a.append(arr_a)
        pooled_b.append(arr_b)

    pooled_r2, pooled_rmse = _r2_rmse(np.concatenate(pooled_a), np.concatenate(pooled_b))
    return AgreementResult(n_pixels=n_pixels, bands=band_results, pooled_r2=pooled_r2, pooled_rmse=pooled_rmse)
