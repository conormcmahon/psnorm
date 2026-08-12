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

Everything here streams over row-blocks of a raster *window* (not
necessarily the full scene — see io.overlap_window) so memory use is bounded
by block_rows * width * bands, independent of how large the overlap is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy import stats

from . import io


class RunningWeightedCovariance:
    """Single-pass weighted mean/covariance accumulator (West, 1979).

    Given a stream of batches (X_i, w_i), tracks the weighted mean and
    weighted sum-of-squares-and-cross-products (SSCP) without ever holding
    the full stream in memory. `covariance()` returns SSCP / (sum(w) - 1).
    """

    def __init__(self, n_vars: int):
        self.n_vars = n_vars
        self.reset()

    def reset(self):
        self.mean = np.zeros(self.n_vars)
        self.sscp = np.zeros((self.n_vars, self.n_vars))
        self.sum_weights = 1e-9  # avoids divide-by-zero before any data arrives

    def update(self, X: np.ndarray, weights: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] == 0:
            return
        if weights is None:
            weights = np.ones(X.shape[0])
        new_sum_weights = self.sum_weights + weights.sum()
        delta_old = X - self.mean
        weighted_delta = weights[:, None] * delta_old
        self.mean = self.mean + weighted_delta.sum(axis=0) / new_sum_weights
        delta_new = X - self.mean
        self.sscp = self.sscp + weighted_delta.T @ delta_new
        self.sum_weights = new_sum_weights

    def covariance(self) -> np.ndarray:
        cov = self.sscp / (self.sum_weights - 1.0)
        return 0.5 * (cov + cov.T)  # symmetrize accumulated round-off drift

    def means(self) -> np.ndarray:
        return self.mean


def _solve_canonical_correlation(s11: np.ndarray, s22: np.ndarray, s12: np.ndarray):
    """Canonical correlations rho and canonical-variate weight matrices A, B
    for reference-band block s11, target-band block s22, cross-block s12.

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


def _log_transform_tile(tile: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(tile, _LOG_TRANSFORM_EPS))


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


def mad_chisqr(tile_ref: np.ndarray, tile_tgt: np.ndarray, fit: IrMadFit) -> np.ndarray:
    """Per-pixel chi-square statistic of the MAD variates for a
    (n_pixels, n_bands) tile pair, given a fitted IrMadFit."""
    mads = (tile_ref - fit.means1) @ fit.A - (tile_tgt - fit.means2) @ fit.B
    return np.sum((mads / fit.sigma) ** 2, axis=1)


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
) -> IrMadFit:
    """Fit IR-MAD over the given reference/target windows, streamed in
    row-blocks. `valid_mask` is a boolean (ysize, xsize) array (matching the
    window size) excluding cloud/shadow/nodata pixels from the covariance —
    computed once by the caller (see masking.build_validity_mask), not
    recomputed per iteration.

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

    ref_ds, ref_bands = io.open_bands(ref_path, band_indices_ref)
    tgt_ds, tgt_bands = io.open_bands(tgt_path, band_indices_tgt)
    rxoff, ryoff, xsize, ysize = ref_window
    txoff, tyoff, _, _ = tgt_window

    cov = RunningWeightedCovariance(2 * n_bands)
    old_rho = np.zeros(n_bands)
    fit: IrMadFit | None = None
    history: list[tuple[float, IrMadFit]] = []
    delta_threshold = 1.0 - conv_threshold

    n_iterations = 0
    while n_iterations < max_iter:
        try:
            for row_off, n_rows in io.iter_row_blocks(ysize, block_rows):
                tile_ref_raw = io.read_block_flat(ref_bands, rxoff, ryoff + row_off, xsize, n_rows, downsample_factor)
                tile_tgt_raw = io.read_block_flat(tgt_bands, txoff, tyoff + row_off, xsize, n_rows, downsample_factor)
                mask_block = valid_mask[row_off : row_off + n_rows, :].ravel()
                keep = mask_block & tile_ref_raw.any(axis=1) & tile_tgt_raw.any(axis=1)

                if log_transform:
                    tile_ref = _log_transform_tile(tile_ref_raw)
                    tile_tgt = _log_transform_tile(tile_tgt_raw)
                else:
                    tile_ref, tile_tgt = tile_ref_raw, tile_tgt_raw

                if fit is not None:
                    chisqr = mad_chisqr(tile_ref, tile_tgt, fit)
                    weights = stats.chi2.sf(chisqr, n_bands)
                    cov.update(np.concatenate([tile_ref[keep], tile_tgt[keep]], axis=1), weights[keep])
                else:
                    cov.update(np.concatenate([tile_ref[keep], tile_tgt[keep]], axis=1))

            S = cov.covariance()
            means = cov.means()
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

    ref_ds = tgt_ds = None

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
    return fit
