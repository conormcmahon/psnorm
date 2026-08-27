"""IR-MAD: Iteratively Re-weighted Multivariate Alteration Detection.

Clean-room implementation of the published algorithm (Nielsen, 2007, "The
Regularized Iteratively Reweighted MAD Method for Change Detection in
Multi- and Hyperspectral Data"; Canty & Nielsen, 2008): find the linear
combinations of the reference and target bands that are maximally
correlated (canonical correlation analysis on the stacked band covariance),
whose *differences* (the MAD variates) then have minimum correlation and
maximum discriminating power for change. Pixels are then reweighted each
iteration by their chi-square no-change probability under the MAD variates,
so the covariance used for the next canonical-correlation solve is
increasingly dominated by genuinely invariant pixels.

The reference/target *window* (not necessarily the full scene — see
io.overlap_window) is read from disk exactly once, in row-block chunks (so
peak memory during the read itself stays bounded by block_rows * width *
bands), and then kept resident in memory (or on-device, see `device` below)
for however many IR-MAD iterations it takes to converge (up to `max_iter`,
default 30) — the iteration loop itself never touches disk again. This
matters more than it might look: IR-MAD's own reweighting scheme requires
re-scoring the *same* pixels against an updated model every iteration, so
without this caching the whole window would otherwise be re-read (and, for
compressed source rasters, re-decompressed) from disk up to `max_iter`
times per scene.

`device` ("cpu", "gpu", or "auto" — see psnorm.backend) controls which array
backend (NumPy or CuPy) the large (n_pixels, n_bands) per-iteration math
runs on. The canonical-correlation eigensolve itself (`_solve_canonical_
correlation`) and the chi-square reweighting (`scipy.stats.chi2.sf`) always
run on host regardless of `device`: the eigenproblem is on a tiny B x B
matrix (B = band count, 4-8) where GPU kernel-launch latency would dominate
any real compute, and `scipy.stats` has no reliably-available GPU
equivalent. Only the bulk elementwise/matmul work (log-transform, the MAD
chi-square statistic, the weighted-covariance accumulation) runs on
`device` — small arrays cross the host/device boundary each iteration,
large ones never do after the initial cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy import stats

from . import backend, io


class RunningWeightedCovariance:
    """Single-pass weighted mean/covariance accumulator (West, 1979).

    Given a stream of batches (X_i, w_i), tracks the weighted mean and
    weighted sum-of-squares-and-cross-products (SSCP) without ever holding
    the full stream in memory. `covariance()` returns SSCP / (sum(w) - 1).

    `xp` (default `numpy`) is the array module batches/results are created
    with — pass `cupy` to accumulate entirely on-device.
    """

    def __init__(self, n_vars: int, xp=np):
        self.n_vars = n_vars
        self.xp = xp
        self.reset()

    def reset(self):
        xp = self.xp
        self.mean = xp.zeros(self.n_vars)
        self.sscp = xp.zeros((self.n_vars, self.n_vars))
        self.sum_weights = 1e-9  # avoids divide-by-zero before any data arrives

    def update(self, X, weights=None):
        xp = self.xp
        X = xp.asarray(X, dtype=xp.float64)
        if X.shape[0] == 0:
            return
        if weights is None:
            weights = xp.ones(X.shape[0])
        else:
            weights = xp.asarray(weights, dtype=xp.float64)
        new_sum_weights = self.sum_weights + weights.sum()
        delta_old = X - self.mean
        weighted_delta = weights[:, None] * delta_old
        self.mean = self.mean + weighted_delta.sum(axis=0) / new_sum_weights
        delta_new = X - self.mean
        self.sscp = self.sscp + weighted_delta.T @ delta_new
        self.sum_weights = new_sum_weights

    def covariance(self):
        cov = self.sscp / (self.sum_weights - 1.0)
        return 0.5 * (cov + cov.T)  # symmetrize accumulated round-off drift

    def means(self):
        return self.mean


def _solve_canonical_correlation(s11: np.ndarray, s22: np.ndarray, s12: np.ndarray):
    """Canonical correlations rho and canonical-variate weight matrices A, B
    for reference-band block s11, target-band block s22, cross-block s12.

    Always runs on host NumPy/SciPy regardless of the caller's `device`:
    s11/s22/s12 are B x B (B = band count, typically 4-8), so a GPU solve
    here would be dominated by kernel-launch latency, not compute — callers
    are expected to pass host arrays (see `backend.to_host`).

    Solves the two coupled generalized eigenproblems
        s12 s22^-1 s21 a = mu^2 s11 a
        s21 s11^-1 s12 b = mu^2 s22 b
    and returns rho = sqrt(mu^2) sorted ascending (rho[0] is the *least*
    correlated pair, i.e. the pair with the most change-detection power —
    matching Canty & Nielsen's convention).
    """
    n_bands = s11.shape[0]
    s21 = s12.T
    if n_bands > 1:
        c1 = s12 @ scipy.linalg.solve(s22, s21, assume_a="pos")
        c2 = s21 @ scipy.linalg.solve(s11, s12, assume_a="pos")
        eig_a, A = scipy.linalg.eigh(0.5 * (c1 + c1.T), 0.5 * (s11 + s11.T))
        eig_b, B = scipy.linalg.eigh(0.5 * (c2 + c2.T), 0.5 * (s22 + s22.T))
        order_a, order_b = np.argsort(eig_a), np.argsort(eig_b)
        A, B = A[:, order_a], B[:, order_b]
        mu2 = eig_b[order_b]
    else:
        mu2 = np.array([(s12[0, 0] * s21[0, 0] / s22[0, 0]) / s11[0, 0]])
        A = np.array([[1.0 / np.sqrt(s11[0, 0])]])
        B = np.array([[1.0 / np.sqrt(s22[0, 0])]])

    mu2 = np.clip(mu2, 0.0, 1.0)  # round-off can push mu^2 outside [0, 1]
    rho = np.sqrt(mu2)

    # Sign-fix: without this, eigenvectors can flip sign between iterations
    # (both +v and -v solve the same eigenproblem), which would corrupt the
    # delta-based convergence check.
    d = 1.0 / np.sqrt(np.diag(s11))
    sign_a = np.sign(np.sum(d[:, None] * s11 @ A, axis=0))
    sign_a[sign_a == 0] = 1.0
    A = A * sign_a
    sign_b = np.sign(np.diag(A.T @ s12 @ B))
    sign_b[sign_b == 0] = 1.0
    B = B * sign_b

    return rho, A, B


_LOG_TRANSFORM_EPS = 1e-6  # floor before log() so nodata/near-zero DN never produces -inf/NaN


def _log_transform_tile(tile, xp=np):
    return xp.log(xp.maximum(tile, _LOG_TRANSFORM_EPS))


@dataclass
class IrMadFit:
    rho: np.ndarray       # canonical correlations, shape (n_bands,)
    A: np.ndarray          # reference canonical weights, (n_bands, n_bands)
    B: np.ndarray           # target canonical weights, (n_bands, n_bands)
    means1: np.ndarray       # reference band means at convergence, (n_bands,)
    means2: np.ndarray        # target band means at convergence, (n_bands,)
    sigma: np.ndarray          # MAD variate std devs, sqrt(2*(1-rho)), (n_bands,)
    n_iterations: int
    converged: bool
    # Every array field above is always plain host NumPy, regardless of what
    # `device` fit_irmad ran on — they're tiny (O(n_bands^2) at most), so
    # keeping them backend-agnostic makes IrMadFit trivially JSON-
    # serializable (see model_io.save_irmad_fit) and reusable by
    # mad_chisqr() against tiles on *either* device without extra plumbing.


def mad_chisqr(tile_ref, tile_tgt, fit: IrMadFit, xp=np) -> np.ndarray:
    """Per-pixel chi-square statistic of the MAD variates for a
    (n_pixels, n_bands) tile pair, given a fitted IrMadFit.

    `tile_ref`/`tile_tgt` may be NumPy or CuPy arrays (whichever `xp` is);
    `fit`'s fields are always host NumPy (see IrMadFit) and are moved onto
    `xp` here on the fly — cheap, since they're tiny."""
    A = xp.asarray(fit.A)
    B = xp.asarray(fit.B)
    means1 = xp.asarray(fit.means1)
    means2 = xp.asarray(fit.means2)
    sigma = xp.asarray(fit.sigma)
    mads = (tile_ref - means1) @ A - (tile_tgt - means2) @ B
    return xp.sum((mads / sigma) ** 2, axis=1)


@dataclass
class CachedWindow:
    """The reference/target window's pixel data, read from disk exactly
    once (see `_load_window_cache`) and reused across every IR-MAD
    iteration and the final invariant-pixel classification pass — the
    thing that lets both of those skip re-reading (and, for compressed
    source rasters, re-decompressing) the same pixels over and over.

    `blocks` is a list of (row_off, n_rows, tile_ref, tile_tgt, keep)
    tuples, one per row-block (matching io.iter_row_blocks) — kept
    block-chunked (not concatenated into one array) purely so the read
    itself stays bounded by block_rows * width * bands rather than needing
    the whole window materialized to build one contiguous array; every
    consumer just iterates the list. `tile_ref`/`tile_tgt` are already
    log-transformed if `log_transform` was requested (computing that once
    here, rather than once per iteration, is free — the transform is a
    pure function of the raw tile and doesn't depend on iteration state).
    All arrays live on `device`.
    """

    blocks: list
    device: str


def _load_window_cache(
    ref_bands, tgt_bands, rxoff, ryoff, txoff, tyoff, xsize, ysize,
    valid_mask: np.ndarray, *, block_rows: int, downsample_factor: int,
    log_transform: bool, xp,
) -> CachedWindow:
    # Computed once from the *reference* window and reused for both the
    # reference and target reads of every block, so every target scene's
    # coarse-cell grid lands at the same absolute positions on the
    # reference grid (see io.read_block_flat's docstring) -- not reused
    # independently per-raster, which would phase-shift each target's grid
    # by a different, essentially arbitrary amount.
    x_phase = rxoff % downsample_factor if downsample_factor > 1 else 0
    y_phase = ryoff % downsample_factor if downsample_factor > 1 else 0

    blocks = []
    for row_off, n_rows in io.iter_row_blocks(ysize, block_rows):
        tile_ref_raw = io.read_block_flat(
            ref_bands, rxoff, ryoff + row_off, xsize, n_rows, downsample_factor, x_phase, y_phase,
        )
        tile_tgt_raw = io.read_block_flat(
            tgt_bands, txoff, tyoff + row_off, xsize, n_rows, downsample_factor, x_phase, y_phase,
        )
        mask_block = valid_mask[row_off : row_off + n_rows, :].ravel()
        keep = mask_block & tile_ref_raw.any(axis=1) & tile_tgt_raw.any(axis=1)

        if log_transform:
            tile_ref = _log_transform_tile(tile_ref_raw)
            tile_tgt = _log_transform_tile(tile_tgt_raw)
        else:
            tile_ref, tile_tgt = tile_ref_raw, tile_tgt_raw

        if xp is not np:
            tile_ref = xp.asarray(tile_ref)
            tile_tgt = xp.asarray(tile_tgt)
            keep = xp.asarray(keep)
        blocks.append((row_off, n_rows, tile_ref, tile_tgt, keep))

    return CachedWindow(blocks=blocks, device="gpu" if xp is not np else "cpu")


def fit_irmad(
    ref_path: str,
    tgt_path: str,
    ref_window: io.Window,
    tgt_window: io.Window,
    band_indices_ref: list[int],
    band_indices_tgt: list[int],
    valid_mask: np.ndarray,
    *,
    max_iter: int = 30,
    conv_threshold: float = 0.99,
    block_rows: int = io.DEFAULT_BLOCK_ROWS,
    downsample_factor: int = 1,
    log_transform: bool = True,
    device: str = "cpu",
    return_cache: bool = False,
    _cache: CachedWindow | None = None,
):
    """Fit IR-MAD over the given reference/target windows. `valid_mask` is a
    boolean (ysize, xsize) array (matching the window size) excluding
    cloud/shadow/nodata pixels from the covariance — computed once by the
    caller (see masking.build_validity_mask), not recomputed per iteration.

    `device` ("cpu"/"gpu"/"auto") selects the array backend for the large
    per-pixel math — see psnorm.backend and this module's docstring for what
    stays on host regardless. Must already be *resolved* ("cpu" or "gpu",
    not "auto" — see backend.resolve_device); resolving per-call here would
    mean re-probing for a GPU on every one of potentially thousands of
    scenes instead of once per pipeline run.

    `return_cache=True` additionally returns the `CachedWindow` the window
    was read into, so a caller that also needs to classify pixels against
    the converged fit (see normalize.detect_invariant_candidates) can reuse
    the same in-memory/on-device data instead of reading the window again.
    `_cache` lets a caller that already *has* a CachedWindow (e.g. from a
    previous call against the same window) skip the read entirely — used by
    the fast re-classification path (normalize.reclassify_invariant_pixels)
    that reuses a saved fit's own cache-free window read.

    `downsample_factor` > 1 fits against spatially-coarsened pixel values
    (see io.read_block_flat) to reduce sensitivity to per-pixel sensor/
    registration noise when searching for invariant targets — window/mask
    shapes are completely unaffected, only the values read from disk are
    coarsened. block_rows is snapped down to a multiple of
    downsample_factor so each streamed block downsamples cleanly.

    `log_transform` (default True) fits against log(DN) rather than raw DN.
    Sensor noise and real BRDF/illumination-driven fluctuation both scale
    roughly with signal level (heteroscedastic in DN space), so a fixed
    absolute-DN chi-square criterion systematically penalizes bright
    surfaces for showing proportionally-larger absolute swings even when
    they're just as *relatively* stable as dark ones — biasing invariant-
    target selection toward dark, low-DN pixels regardless of their true
    temporal stability. Log-DN differences are proportional (percentage)
    differences, which puts bright and dark surfaces on a comparable
    footing. Values are floored at a small epsilon before the log so
    nodata/near-zero DN can't produce -inf/NaN (those pixels are excluded
    from the fit by `valid_mask` regardless).
    """
    n_bands = len(band_indices_ref)
    if len(band_indices_tgt) != n_bands:
        raise ValueError("Reference and target band counts must match.")
    if downsample_factor > 1:
        block_rows = max(downsample_factor, (block_rows // downsample_factor) * downsample_factor)

    xp = backend.get_array_module(device)
    rxoff, ryoff, xsize, ysize = ref_window
    txoff, tyoff, _, _ = tgt_window

    if _cache is not None:
        cache = _cache
    else:
        ref_ds, ref_bands = io.open_bands(ref_path, band_indices_ref)
        tgt_ds, tgt_bands = io.open_bands(tgt_path, band_indices_tgt)
        cache = _load_window_cache(
            ref_bands, tgt_bands, rxoff, ryoff, txoff, tyoff, xsize, ysize, valid_mask,
            block_rows=block_rows, downsample_factor=downsample_factor,
            log_transform=log_transform, xp=xp,
        )
        ref_ds = tgt_ds = None  # datasets only needed for the one-time read above

    cov = RunningWeightedCovariance(2 * n_bands, xp=xp)
    old_rho = np.zeros(n_bands)
    fit: IrMadFit | None = None
    history: list[tuple[float, IrMadFit]] = []
    delta_threshold = 1.0 - conv_threshold

    n_iterations = 0
    while n_iterations < max_iter:
        try:
            for row_off, n_rows, tile_ref, tile_tgt, keep in cache.blocks:
                if fit is not None:
                    chisqr = mad_chisqr(tile_ref, tile_tgt, fit, xp=xp)
                    weights_host = stats.chi2.sf(backend.to_host(chisqr), n_bands)
                    weights = xp.asarray(weights_host) if xp is not np else weights_host
                    cov.update(xp.concatenate([tile_ref[keep], tile_tgt[keep]], axis=1), weights[keep])
                else:
                    cov.update(xp.concatenate([tile_ref[keep], tile_tgt[keep]], axis=1))

            S = backend.to_host(cov.covariance())
            means = backend.to_host(cov.means())
            cov.reset()

            s11 = S[:n_bands, :n_bands]
            s22 = S[n_bands:, n_bands:]
            s12 = S[:n_bands, n_bands:]
            rho, A, B = _solve_canonical_correlation(s11, s22, s12)
            sigma = np.sqrt(2.0 * (1.0 - rho))

            n_iterations += 1
            fit = IrMadFit(
                rho=rho,
                A=A,
                B=B,
                means1=means[:n_bands],
                means2=means[n_bands:],
                sigma=sigma,
                n_iterations=n_iterations,
                converged=False,
            )
            delta = float(np.max(np.abs(rho - old_rho)))
            old_rho = rho
            history.append((delta, fit))

            if n_iterations > 1 and delta < delta_threshold:
                fit.converged = True
                break
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
            break

    if not history:
        raise RuntimeError(
            f"IR-MAD failed on its first iteration fitting '{tgt_path}' "
            f"against '{ref_path}' (singular covariance — check for "
            f"degenerate/constant bands in the overlap region)."
        )
    if fit is None or not fit.converged:
        # Max iterations reached (or a later iteration errored): fall back
        # to the iteration with the smallest delta seen so far.
        _, fit = min(history, key=lambda h: h[0])

    if return_cache:
        return fit, cache
    return fit
