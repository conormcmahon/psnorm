"""Invariant-pixel selection (chi-square no-change probability on the IR-MAD
statistic) and per-band orthogonal regression, producing a NormalizationModel.

The regression pass streams over the same row-blocks as irmad.fit_irmad
rather than materializing the selected pixels as arrays: each block updates
a small (n, sum_x, sum_y, sum_x2, sum_y2, sum_xy) accumulator per band, and
the closed-form total-least-squares (orthogonal regression) solution is
computed once from those accumulated sums at the end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy import stats

from . import backend, io, irmad, masking


def select_invariant_pixels(chisqr: np.ndarray, dof: int, ncp_threshold: float) -> np.ndarray:
    """Boolean mask of pixels whose no-change probability (under the
    chi-square distribution with `dof` degrees of freedom) exceeds
    `ncp_threshold`."""
    ncp = stats.chi2.sf(chisqr, dof)
    return ncp > ncp_threshold


class _RegressionAccumulator:
    """Streaming sufficient statistics for closed-form orthogonal
    (total-least-squares) regression of y on x."""

    def __init__(self):
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n += x.size
        self.sum_x += x.sum()
        self.sum_y += y.sum()
        self.sum_x2 += float(np.sum(x * x))
        self.sum_y2 += float(np.sum(y * y))
        self.sum_xy += float(np.sum(x * y))

    def finalize(self) -> tuple[float, float, float, int]:
        """Returns (slope, intercept, r_squared, n)."""
        if self.n < 2:
            raise ValueError(
                f"Only {self.n} invariant pixels available — too few to "
                f"fit a regression. Lower ncp_threshold or check the input "
                f"data/masking for this scene pair."
            )
        mean_x = self.sum_x / self.n
        mean_y = self.sum_y / self.n
        dof = self.n - 1
        sxx = (self.sum_x2 - self.n * mean_x**2) / dof
        syy = (self.sum_y2 - self.n * mean_y**2) / dof
        sxy = (self.sum_xy - self.n * mean_x * mean_y) / dof

        if sxy == 0.0:
            return 0.0, mean_y, 0.0, self.n

        denom = math.sqrt(max(sxx * syy, 0.0))
        r = sxy / denom if denom > 0 else 0.0
        slope = (syy - sxx + math.sqrt((syy - sxx) ** 2 + 4.0 * sxy * sxy)) / (2.0 * sxy)
        intercept = mean_y - slope * mean_x
        return slope, intercept, r * r, self.n


def fit_orthogonal_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Orthogonal (total-least-squares) regression of y on x over full
    in-memory arrays. Convenience wrapper around _RegressionAccumulator for
    callers/tests that already have the data in memory; the real pipeline
    uses the streaming accumulator directly. Returns (slope, intercept, r2).
    """
    acc = _RegressionAccumulator()
    acc.update(x, y)
    slope, intercept, r2, _n = acc.finalize()
    return slope, intercept, r2


@dataclass
class BandModel:
    band_name: str
    slope: float
    intercept: float
    r2: float
    n_invariant_pixels: int
    n_outliers_excluded: int = 0
    rmse_before: float = float("nan")  # target vs reference, at this band's fitted points, before correction
    rmse_after: float = float("nan")   # target (corrected by the *fitted* slope/intercept, i.e. raw_slope/
                                        # raw_intercept when a fallback happened) vs reference, same points --
                                        # describes the underlying fit's quality regardless of whether it was
                                        # actually applied; see identity_fallback for what apply.py actually used
    identity_fallback: bool = False    # True if slope/intercept below were replaced with the identity transform
    fallback_reason: str | None = None  # why, e.g. "too few pixels" or "slope outside bounds" -- None if not
    raw_slope: float | None = None      # the fitted (RANSAC) slope/intercept *before* the fallback decision,
    raw_intercept: float | None = None  # only populated when identity_fallback is True (else redundant with slope/intercept)


@dataclass
class NormalizationModel:
    reference_id: str
    target_id: str
    band_names: list[str]
    bands: list[BandModel]
    n_consensus_pixels: int
    invariance_frequency_threshold: float
    min_observations: int
    fitted_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def band(self, name: str) -> BandModel:
        for b in self.bands:
            if b.band_name == name:
                return b
        raise KeyError(f"No band model for '{name}'")


def identity_model(band_names: list[str], scene_id: str) -> NormalizationModel:
    """A no-op NormalizationModel (slope=1, intercept=0 per band), used for
    the reference scene itself so it can flow through apply.apply_model()
    like every other scene and participate uniformly in the adjacency
    report."""
    bands = [
        BandModel(band_name=name, slope=1.0, intercept=0.0, r2=1.0, n_invariant_pixels=0,
                  rmse_before=0.0, rmse_after=0.0)
        for name in band_names
    ]
    return NormalizationModel(
        reference_id=scene_id,
        target_id=scene_id,
        band_names=list(band_names),
        bands=bands,
        n_consensus_pixels=0,
        invariance_frequency_threshold=float("nan"),
        min_observations=0,
    )


@dataclass
class InvariantDetectionResult:
    """Output of Phase A: this target's candidate invariant pixels vs. the
    reference, not yet checked for cross-scene consensus (see consensus.py)
    and not yet the basis of any regression fit."""

    reference_id: str
    target_id: str
    ref_window: io.Window
    tgt_window: io.Window
    invariant_mask: np.ndarray  # boolean, shape (ysize, xsize) of ref_window
    n_evaluated: int
    n_invariant: int
    ncp_threshold: float
    irmad_rho: list[float]
    irmad_converged: bool
    irmad_iterations: int
    irmad_fit: "irmad.IrMadFit" = None  # the full fitted model -- see model_io.save_irmad_fit


def _classify_from_cache(
    cache: irmad.CachedWindow, fit: irmad.IrMadFit, dof: int, ncp_threshold: float, xsize: int, ysize: int,
) -> tuple[np.ndarray, int, int]:
    """Shared by detect_invariant_candidates (fresh fit) and
    reclassify_invariant_pixels (saved fit, new threshold): score every
    cached block against `fit` and threshold at `ncp_threshold`. Touches no
    disk — `cache` already holds every block's pixel data on whichever
    device it was read to (see irmad._load_window_cache)."""
    xp = backend.get_array_module(cache.device)
    invariant_mask = np.zeros((ysize, xsize), dtype=bool)
    n_evaluated = 0
    n_invariant = 0
    for row_off, n_rows, tile_ref, tile_tgt, keep in cache.blocks:
        chisqr = backend.to_host(irmad.mad_chisqr(tile_ref, tile_tgt, fit, xp=xp))
        keep_host = backend.to_host(keep)
        n_evaluated += int(keep_host.sum())
        invariant = keep_host & select_invariant_pixels(chisqr, dof, ncp_threshold)
        n_invariant += int(invariant.sum())
        invariant_mask[row_off : row_off + n_rows, :] = invariant.reshape(n_rows, xsize)
    return invariant_mask, n_evaluated, n_invariant


def detect_invariant_candidates(
    reference_path: str,
    target_path: str,
    reference_id: str,
    target_id: str,
    ref_window: io.Window,
    tgt_window: io.Window,
    band_indices_ref: list[int],
    band_indices_tgt: list[int],
    band_names: list[str],
    search_mask: np.ndarray,
    *,
    max_iter: int = 30,
    conv_threshold: float = 0.99,
    ncp_threshold: float = 0.70,
    block_rows: int = io.DEFAULT_BLOCK_ROWS,
    downsample_factor: int = 1,
    log_transform: bool = True,
    device: str = "cpu",
) -> InvariantDetectionResult:
    """Fit IR-MAD between reference and target over the given windows, then
    flag pixels whose no-change probability exceeds `ncp_threshold` as
    invariant *candidates*. `search_mask` (nodata/water/cloud excluded,
    see masking.py) gates which pixels IR-MAD is fit on and which pixels are
    eligible to be flagged at all — masked pixels are never candidates.

    The classification pass reuses the exact window data `irmad.fit_irmad`
    already read for the fit (via `return_cache=True`) rather than reading
    the window from disk a second time — see irmad.py's module docstring.

    `device` ("cpu"/"gpu"/"auto", already *resolved* — see
    backend.resolve_device) selects the array backend fit_irmad's iteration
    and this function's classification pass run on.

    `downsample_factor` > 1 both fits IR-MAD and evaluates the chi-square
    statistic against spatially-coarsened pixel values (see
    io.read_block_flat) — the fit and the classification pass must use the
    same coarsening, or the fitted sigma (tuned to the coarser signal's
    noise level) would misjudge finer-grained native noise. The output
    `invariant_mask` is still shaped/indexed exactly like `ref_window` at
    native resolution — only the pixel *values* analyzed are coarsened.

    `log_transform` (default True, see irmad.fit_irmad) must be applied
    identically here and in the fit: chi-square is only meaningful measured
    in whatever space the fit's means/sigma were estimated in.

    This does not fit any regression — see fit_regression_from_mask, called
    later against the cross-scene consensus mask (consensus.py), not this
    per-pair candidate set directly.
    """
    if downsample_factor > 1:
        block_rows = max(downsample_factor, (block_rows // downsample_factor) * downsample_factor)

    fit, cache = irmad.fit_irmad(
        reference_path, target_path, ref_window, tgt_window,
        band_indices_ref, band_indices_tgt, search_mask,
        max_iter=max_iter, conv_threshold=conv_threshold, block_rows=block_rows,
        downsample_factor=downsample_factor, log_transform=log_transform,
        device=device, return_cache=True,
    )

    n_bands = len(band_names)
    dof = n_bands  # number of MAD variates == number of bands
    _, _, xsize, ysize = ref_window

    invariant_mask, n_evaluated, n_invariant = _classify_from_cache(cache, fit, dof, ncp_threshold, xsize, ysize)

    return InvariantDetectionResult(
        reference_id=reference_id,
        target_id=target_id,
        ref_window=ref_window,
        tgt_window=tgt_window,
        invariant_mask=invariant_mask,
        n_evaluated=n_evaluated,
        n_invariant=n_invariant,
        ncp_threshold=ncp_threshold,
        irmad_rho=[float(v) for v in fit.rho],
        irmad_converged=fit.converged,
        irmad_iterations=fit.n_iterations,
        irmad_fit=fit,
    )


def load_reclassification_cache(
    reference_path: str,
    target_path: str,
    ref_window: io.Window,
    tgt_window: io.Window,
    band_indices_ref: list[int],
    band_indices_tgt: list[int],
    search_mask: np.ndarray,
    *,
    block_rows: int = io.DEFAULT_BLOCK_ROWS,
    downsample_factor: int = 1,
    log_transform: bool = True,
    device: str = "cpu",
) -> irmad.CachedWindow:
    """One-time read of a scene's window, for `reclassify_from_cache` calls
    against many different `ncp_threshold` values without re-reading the
    window once per threshold (see scripts/sweep_thresholds.py, which sweeps
    a whole grid of thresholds per scene — reading each scene's window once
    and reusing it, rather than once per threshold, is a 1/n_thresholds
    reduction in disk I/O for the exact same result, and matters more than
    it might look on a slow or flaky filesystem)."""
    if downsample_factor > 1:
        block_rows = max(downsample_factor, (block_rows // downsample_factor) * downsample_factor)

    xp = backend.get_array_module(device)
    rxoff, ryoff, xsize, ysize = ref_window
    txoff, tyoff, _, _ = tgt_window

    ref_ds, ref_bands = io.open_bands(reference_path, band_indices_ref)
    tgt_ds, tgt_bands = io.open_bands(target_path, band_indices_tgt)
    cache = irmad._load_window_cache(
        ref_bands, tgt_bands, rxoff, ryoff, txoff, tyoff, xsize, ysize, search_mask,
        block_rows=block_rows, downsample_factor=downsample_factor,
        log_transform=log_transform, xp=xp,
    )
    ref_ds = tgt_ds = None
    return cache


def reclassify_from_cache(
    cache: irmad.CachedWindow, fit: irmad.IrMadFit, band_indices_ref: list[int], ncp_threshold: float,
    xsize: int, ysize: int,
) -> tuple[np.ndarray, int, int]:
    """Classify against an already-loaded `cache` (see
    load_reclassification_cache) at a given `ncp_threshold` — no disk I/O,
    just the O(pixels) chi-square/threshold pass. Returns (invariant_mask,
    n_evaluated, n_invariant), same as detect_invariant_candidates reports."""
    dof = len(band_indices_ref)
    return _classify_from_cache(cache, fit, dof, ncp_threshold, xsize, ysize)


def reclassify_invariant_pixels(
    reference_path: str,
    target_path: str,
    ref_window: io.Window,
    tgt_window: io.Window,
    band_indices_ref: list[int],
    band_indices_tgt: list[int],
    search_mask: np.ndarray,
    fit: irmad.IrMadFit,
    *,
    ncp_threshold: float,
    block_rows: int = io.DEFAULT_BLOCK_ROWS,
    downsample_factor: int = 1,
    log_transform: bool = True,
    device: str = "cpu",
) -> tuple[np.ndarray, int, int]:
    """Single-shot convenience wrapper: read a scene's window once (see
    load_reclassification_cache) and classify it once (see
    reclassify_from_cache) at one `ncp_threshold`. A caller sweeping *many*
    thresholds for the same scene should call those two directly instead —
    calling this function once per threshold re-reads the window every time.

    Only one fresh pass over the window is needed (to get pixel values —
    the fit itself isn't persisted alongside the raw data, only its small
    fitted parameters are), so this is O(pixels), not O(pixels *
    iterations): the expensive iterative covariance/eigensolve refinement
    that dominates fit_irmad's cost never runs here.
    """
    cache = load_reclassification_cache(
        reference_path, target_path, ref_window, tgt_window, band_indices_ref, band_indices_tgt, search_mask,
        block_rows=block_rows, downsample_factor=downsample_factor, log_transform=log_transform, device=device,
    )
    _, _, xsize, ysize = ref_window
    return reclassify_from_cache(cache, fit, band_indices_ref, ncp_threshold, xsize, ysize)


_RANSAC_MIN_PREDICTION_MAGNITUDE = 1.0  # DN floor on |predicted| before dividing, avoids blowup near 0


def _ransac_relative_inliers(
    x: np.ndarray, y: np.ndarray, slope: float, intercept: float, outlier_relative_threshold: float,
) -> np.ndarray:
    """Boolean inlier mask: True where the model's predicted reference value
    (intercept + slope*target) is within `outlier_relative_threshold` (a
    fraction, e.g. 0.05 == 5%) of the observed reference value, relative to
    the *prediction* — not a fixed absolute DN difference. `predicted` is
    floored in magnitude before dividing so a near-zero prediction can't
    produce a spurious huge relative error.
    """
    predicted = intercept + slope * x
    denom = np.maximum(np.abs(predicted), _RANSAC_MIN_PREDICTION_MAGNITUDE)
    return np.abs(y - predicted) / denom <= outlier_relative_threshold


def _ransac_orthogonal_regression(
    x: np.ndarray,
    y: np.ndarray,
    outlier_relative_threshold: float,
    *,
    max_iterations: int = 300,
    sample_size: int = 12,
    seed: int = 0,
    slope_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """RANSAC-selected inlier (x, y) subset for orthogonal regression.

    Repeatedly fits a candidate orthogonal-regression line from a small
    random subset of points, scores each candidate by how many of *all*
    the points it explains within `outlier_relative_threshold` (a relative,
    percentage-of-prediction threshold — see _ransac_relative_inliers, not
    a fixed absolute DN difference, so the same threshold is equally
    strict for dim and bright targets), and keeps whichever candidate model
    has the most support. The winning inlier set is then refit once and
    re-scored against that refined model (a standard RANSAC "polish" step),
    so the minimal-sample candidate doesn't itself become the final answer.

    `slope_bounds`, if given, is enforced *during the search itself*: a
    sampled candidate whose own slope falls outside `slope_bounds` is
    skipped outright — never scored, never eligible to become `best_mask` —
    rather than being allowed to win purely on inlier count. Without this,
    a scene where cloud-shadow/vegetation-contaminated points outnumber the
    genuine invariant targets among the consensus set can converge on an
    implausible-but-well-supported model (e.g. slope near 0), which then
    gets rejected by the caller's post-hoc plausibility check anyway (see
    fit_regression_from_mask) — wasting the whole search on an answer that
    was never going to be used instead of continuing to look for a
    plausible one. The polish step is gated the same way: if refitting on
    the winning inlier set would push the slope back out of bounds, the
    polish is discarded and the pre-polish (already-plausible) inliers are
    kept as the answer instead.

    Returns `(x_trimmed, y_trimmed, n_excluded, inlier_mask)` — `inlier_mask`
    is a boolean array the same length/order as the *input* x/y, True where
    that original point survived as a final inlier (so a caller that also
    has each point's spatial location can recover which pixels were used,
    e.g. to save a diagnostic raster — see fit_regression_from_mask's
    `return_inlier_masks`). The caller does its own final
    fit_orthogonal_regression on `x_trimmed`/`y_trimmed` to get the reported
    model. Falls back to no trimming (`inlier_mask` all True) if there are
    too few points, or if no sampled candidate is ever both workable and
    (when `slope_bounds` is given) plausible.
    """
    n = x.size
    if n < max(6, sample_size):
        return x, y, 0, np.ones(n, dtype=bool)

    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = -1

    for _ in range(max_iterations):
        idx = rng.choice(n, size=sample_size, replace=False)
        try:
            slope, intercept, _r2 = fit_orthogonal_regression(x[idx], y[idx])
        except (ValueError, ZeroDivisionError):
            continue
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            continue
        if slope_bounds is not None and not (slope_bounds[0] <= slope <= slope_bounds[1]):
            continue
        inliers = _ransac_relative_inliers(x, y, slope, intercept, outlier_relative_threshold)
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_mask = inliers

    # A winning candidate supported by fewer points than it took to define
    # it (< sample_size) isn't meaningfully "well-supported" -- 2 points
    # trivially fit a line with r2=1 regardless of how nonsensical the
    # resulting slope is, so a tiny winning inlier count is a red flag, not
    # a good answer. This matters more once slope_bounds narrows the
    # eligible-candidate pool: with fewer candidates competing, a small,
    # coincidentally-plausible-looking cluster is more likely to end up
    # "best" by default. Falling back to no-trimming here defers to the
    # caller's own min_fit_pixels/slope_bounds post-hoc check (see
    # fit_regression_from_mask), same as the "no candidate ever plausible"
    # case already does.
    if best_mask is None or best_count < sample_size:
        return x, y, 0, np.ones(n, dtype=bool)

    # Polish: refit on the winning inlier set, then re-score everyone
    # against that refined (not minimal-sample) model -- but only if the
    # polished slope is still plausible; otherwise keep the pre-polish
    # (already-plausible) inliers rather than let the polish drift out of
    # bounds.
    slope1, intercept1, _r2_1 = fit_orthogonal_regression(x[best_mask], y[best_mask])
    if (
        np.isfinite(slope1) and np.isfinite(intercept1)
        and (slope_bounds is None or (slope_bounds[0] <= slope1 <= slope_bounds[1]))
    ):
        polished_mask = _ransac_relative_inliers(x, y, slope1, intercept1, outlier_relative_threshold)
        if polished_mask.sum() >= sample_size:
            best_mask = polished_mask

    return x[best_mask], y[best_mask], int((~best_mask).sum()), best_mask


def fit_regression_from_mask(
    reference_path: str,
    target_path: str,
    reference_id: str,
    target_id: str,
    ref_window: io.Window,
    tgt_window: io.Window,
    band_indices_ref: list[int],
    band_indices_tgt: list[int],
    band_names: list[str],
    consensus_mask: np.ndarray,
    *,
    invariance_frequency_threshold: float,
    min_observations: int,
    block_rows: int = io.DEFAULT_BLOCK_ROWS,
    downsample_factor: int = 1,
    outlier_relative_threshold: float | None = 0.05,
    slope_bounds: tuple[float, float] | None = (0.8, 1.2),
    min_fit_pixels: int = 500,
    return_inlier_masks: bool = False,
) -> NormalizationModel | tuple[NormalizationModel, dict[str, np.ndarray]]:
    """Per-band orthogonal regression (target -> reference), using only the
    pixels flagged True in `consensus_mask` — the final, cross-scene-checked
    invariant target set (see consensus.py), not this pair's own IR-MAD
    output. Streamed over row-blocks; `consensus_mask` must already be
    cropped to `ref_window`'s shape.

    `downsample_factor` > 1 fits the correction factors themselves against
    spatially-coarsened pixel values (see io.read_block_flat), matching
    whatever coarsening was used to find the invariant targets in the first
    place — window/mask shapes are unaffected either way.

    `outlier_relative_threshold` (default 0.05, i.e. 5%; pass None to
    disable) drives a RANSAC pass per band (see _ransac_orthogonal_
    regression): still an orthogonal-regression fit (neither image is
    ground truth — both carry error, so a total-least-squares model form
    is kept rather than switching to OLS), but the outlier criterion is now
    *relative* to the model's own prediction rather than a fixed DN
    difference, so the same threshold is equally strict for a dim road and
    a bright roof.

    `slope_bounds` is also passed *into* the RANSAC search itself (see
    _ransac_orthogonal_regression): a candidate model is only eligible to
    win the search if its own slope is already plausible, so contamination
    (cloud shadow, vegetation, misregistration) that outnumbers genuine
    invariant targets among the consensus set can't win by inlier count
    alone and silently produce an implausible-but-well-supported model —
    RANSAC keeps searching for a plausible one instead.

    After RANSAC, each band's fit is sanity-checked once more before it's
    allowed to be applied: if fewer than `min_fit_pixels` points survived
    RANSAC, or the fitted slope still falls outside `slope_bounds` (either
    check disabled by passing None) — which can still happen if RANSAC's
    search never found *any* plausible candidate and fell back to the
    untrimmed set — the band falls back to the identity transform
    (slope=1.0, intercept=0.0) rather than applying an implausible
    correction. `identity_fallback`/`fallback_reason` on the resulting
    BandModel record when and why this happened; `raw_slope`/`raw_intercept`
    keep the rejected fit around for inspection. `r2`/`rmse_before`/
    `rmse_after` always describe the underlying *fitted* model's quality
    (raw_slope/raw_intercept when a fallback occurred) — they're a
    diagnostic of the fit itself, not of what was actually applied; check
    `identity_fallback` for that. n_invariant_pixels/n_outliers_excluded
    describe the post-RANSAC point set regardless of whether the fit was
    ultimately applied.

    `return_inlier_masks=True` additionally returns a `{band_name: mask}`
    dict, one boolean (ysize, xsize) array per band (shaped like
    `ref_window`, same convention as `consensus_mask`) marking which
    consensus pixels survived as RANSAC's *final* inlier set for that band
    — saved regardless of whether the band was ultimately accepted or fell
    back to identity, so a rejected band's mask is still useful for seeing
    what RANSAC found (e.g. a small, spatially clustered inlier set is a
    different failure mode than a large-but-implausible one). When RANSAC
    is disabled (`outlier_relative_threshold=None`), the mask is just the
    whole consensus set for that band.
    """
    if downsample_factor > 1:
        block_rows = max(downsample_factor, (block_rows // downsample_factor) * downsample_factor)

    n_bands = len(band_names)
    ref_ds, ref_bands = io.open_bands(reference_path, band_indices_ref)
    tgt_ds, tgt_bands = io.open_bands(target_path, band_indices_tgt)
    rxoff, ryoff, xsize, ysize = ref_window
    txoff, tyoff, _, _ = tgt_window

    # Per-band raw (target, reference) value pairs at consensus locations --
    # bounded by the (deliberately sparse) consensus pixel count, not scene
    # size, so materializing them in memory (rather than only streaming
    # sufficient statistics) is cheap and lets outlier removal see actual
    # residuals/robust scale, not just accumulated sums.
    tgt_values: list[list[np.ndarray]] = [[] for _ in range(n_bands)]
    ref_values: list[list[np.ndarray]] = [[] for _ in range(n_bands)]
    # Flat (row*xsize + col) position within ref_window for every collected
    # point -- shared across all bands (built from the same `invariant`
    # selection each block), only materialized when the caller wants inlier
    # rasters back. Lets a band's final RANSAC inlier_mask (an index into
    # this same per-band point ordering) be mapped back to actual pixel
    # locations for return_inlier_masks.
    positions: list[np.ndarray] = []
    n_consensus_pixels = 0

    # See irmad._load_window_cache's identical pattern: computed once from
    # the reference window so every target scene's coarse-cell grid lands
    # at the same absolute positions on the reference grid, not phased
    # independently per scene (see io.read_block_flat's docstring).
    x_phase = rxoff % downsample_factor if downsample_factor > 1 else 0
    y_phase = ryoff % downsample_factor if downsample_factor > 1 else 0

    for row_off, n_rows in io.iter_row_blocks(ysize, block_rows):
        invariant = consensus_mask[row_off : row_off + n_rows, :].ravel()
        if not invariant.any():
            continue
        tile_ref = io.read_block_flat(
            ref_bands, rxoff, ryoff + row_off, xsize, n_rows, downsample_factor, x_phase, y_phase,
        )
        tile_tgt = io.read_block_flat(
            tgt_bands, txoff, tyoff + row_off, xsize, n_rows, downsample_factor, x_phase, y_phase,
        )
        n_consensus_pixels += int(invariant.sum())
        if return_inlier_masks:
            block_start = row_off * xsize
            positions.append(np.arange(block_start, block_start + n_rows * xsize)[invariant])
        for b in range(n_bands):
            # target is x, reference is y: fits reference = a + b*target,
            # the transform later applied to normalize target scenes.
            tgt_values[b].append(tile_tgt[invariant, b])
            ref_values[b].append(tile_ref[invariant, b])

    ref_ds = tgt_ds = None

    if n_consensus_pixels < 2:
        raise ValueError(
            f"Only {n_consensus_pixels} consensus invariant pixels available "
            f"for target='{target_id}' vs reference='{reference_id}' "
            f"(invariance_frequency_threshold={invariance_frequency_threshold}, "
            f"min_observations={min_observations}). Lower these or check "
            f"overlap/masking for this scene pair."
        )

    position_concat = np.concatenate(positions) if positions else None

    band_models = []
    inlier_rasters: dict[str, np.ndarray] = {}
    for b, name in enumerate(band_names):
        x = np.concatenate(tgt_values[b])
        y = np.concatenate(ref_values[b])

        n_outliers = 0
        inlier_mask = np.ones(x.size, dtype=bool)
        if outlier_relative_threshold is not None:
            x, y, n_outliers, inlier_mask = _ransac_orthogonal_regression(
                x, y, outlier_relative_threshold, slope_bounds=slope_bounds,
            )

        # before/after both computed over the same final (post-trim) point
        # set, so the comparison isolates what the correction itself does
        # rather than conflating it with the effect of dropping outliers.
        rmse_before = float(np.sqrt(np.mean((x - y) ** 2)))
        fitted_slope, fitted_intercept, r2 = fit_orthogonal_regression(x, y)
        rmse_after = float(np.sqrt(np.mean((fitted_intercept + fitted_slope * x - y) ** 2)))

        fallback_reason = None
        if x.size < min_fit_pixels:
            fallback_reason = f"only {x.size} consensus pixels survived RANSAC (< min_fit_pixels={min_fit_pixels})"
        elif slope_bounds is not None and not (slope_bounds[0] <= fitted_slope <= slope_bounds[1]):
            fallback_reason = f"slope {fitted_slope:.4f} outside plausible bounds {slope_bounds}"

        if fallback_reason is not None:
            slope, intercept = 1.0, 0.0
            raw_slope, raw_intercept = fitted_slope, fitted_intercept
        else:
            slope, intercept = fitted_slope, fitted_intercept
            raw_slope = raw_intercept = None

        band_models.append(BandModel(
            band_name=name, slope=slope, intercept=intercept, r2=r2, n_invariant_pixels=int(x.size),
            n_outliers_excluded=n_outliers, rmse_before=rmse_before, rmse_after=rmse_after,
            identity_fallback=fallback_reason is not None, fallback_reason=fallback_reason,
            raw_slope=raw_slope, raw_intercept=raw_intercept,
        ))

        if return_inlier_masks:
            raster = np.zeros((ysize, xsize), dtype=bool)
            inlier_positions = position_concat[inlier_mask]
            raster.ravel()[inlier_positions] = True
            inlier_rasters[name] = raster

    model = NormalizationModel(
        reference_id=reference_id,
        target_id=target_id,
        band_names=list(band_names),
        bands=band_models,
        n_consensus_pixels=n_consensus_pixels,
        invariance_frequency_threshold=invariance_frequency_threshold,
        min_observations=min_observations,
    )
    if return_inlier_masks:
        return model, inlier_rasters
    return model


@dataclass
class BandSpectralCoverage:
    """How well one band's invariant-target pixels span that band's actual
    dynamic range in the scene being corrected. A regression fit from
    targets clustered in a narrow slice of the real reflectance range
    extrapolates badly outside that slice — `variance_ratio` near 0 or
    `coverage_fraction` near 0 both flag that risk."""

    band_name: str
    n_target_pixels: int
    target_variance: float
    whole_image_variance: float
    variance_ratio: float       # target_variance / whole_image_variance
    whole_image_p10: float
    whole_image_p90: float
    target_min: float
    target_max: float
    coverage_fraction: float    # overlap([target_min,target_max],[p10,p90]) / (p90-p10)


def compute_spectral_coverage(
    image_path: str,
    window: io.Window,
    band_indices: list[int],
    band_names: list[str],
    invariant_mask: np.ndarray,
    *,
    block_rows: int = io.DEFAULT_BLOCK_ROWS,
) -> list[BandSpectralCoverage]:
    """Per-band spectral-range coverage of the pixels flagged True in
    `invariant_mask` (shaped like `window`, on `image_path`'s own pixel
    grid) against `image_path`'s *entire* scene.

    Always computed from full-fidelity, native-resolution pixel values
    (independent of whether downsample_factor was used to find the targets
    or fit the correction factors) — this report is meant to honestly
    describe the real reflectance diversity at the chosen locations and in
    the scene being corrected, which coarsened values would understate.
    """
    n_bands = len(band_names)
    xoff, yoff, xsize, ysize = window

    ds, bands = io.open_bands(image_path, band_indices)
    target_values: list[list[np.ndarray]] = [[] for _ in range(n_bands)]
    for row_off, n_rows in io.iter_row_blocks(ysize, block_rows):
        mask_block = invariant_mask[row_off : row_off + n_rows, :].ravel()
        if not mask_block.any():
            continue
        tile = io.read_block_flat(bands, xoff, yoff + row_off, xsize, n_rows)
        for b in range(n_bands):
            target_values[b].append(tile[mask_block, b])
    ds = None

    info = io.get_raster_info(image_path)
    full_window = (0, 0, info.width, info.height)
    whole = io.read_window_bands(image_path, full_window, band_indices)  # (bands, y, x)
    whole_valid = masking.read_nodata_valid_mask(image_path, full_window)

    results = []
    for b, name in enumerate(band_names):
        t_vals = np.concatenate(target_values[b]) if target_values[b] else np.array([])
        whole_vals = whole[b][whole_valid]

        n_t = int(t_vals.size)
        t_var = float(np.var(t_vals, ddof=1)) if n_t >= 2 else float("nan")
        w_var = float(np.var(whole_vals, ddof=1)) if whole_vals.size >= 2 else float("nan")
        ratio = t_var / w_var if w_var else float("nan")

        if whole_vals.size:
            p10, p90 = (float(v) for v in np.percentile(whole_vals, [10, 90]))
        else:
            p10, p90 = float("nan"), float("nan")

        t_min = float(t_vals.min()) if n_t else float("nan")
        t_max = float(t_vals.max()) if n_t else float("nan")

        span = p90 - p10
        if n_t and span > 0:
            overlap = max(0.0, min(t_max, p90) - max(t_min, p10))
            coverage = overlap / span
        else:
            coverage = float("nan")

        results.append(BandSpectralCoverage(
            band_name=name, n_target_pixels=n_t, target_variance=t_var,
            whole_image_variance=w_var, variance_ratio=ratio,
            whole_image_p10=p10, whole_image_p90=p90,
            target_min=t_min, target_max=t_max, coverage_fraction=coverage,
        ))
    return results
