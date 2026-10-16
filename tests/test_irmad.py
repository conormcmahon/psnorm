import os

import numpy as np
from scipy import stats as spstats

from psnorm import io, irmad
from psnorm.irmad import RunningWeightedCovariance, _solve_canonical_correlation

from _raster_utils import write_geotiff


def test_running_weighted_covariance_matches_numpy_unweighted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))

    acc = RunningWeightedCovariance(3)
    acc.update(X[:200])
    acc.update(X[200:])

    np.testing.assert_allclose(acc.means(), X.mean(axis=0), atol=1e-8)
    np.testing.assert_allclose(acc.covariance(), np.cov(X, rowvar=False), atol=1e-8)


def test_running_weighted_covariance_matches_numpy_weighted():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2))
    w = rng.uniform(0.1, 2.0, size=300)

    acc = RunningWeightedCovariance(2)
    acc.update(X[:150], w[:150])
    acc.update(X[150:], w[150:])

    expected_mean = np.average(X, axis=0, weights=w)
    np.testing.assert_allclose(acc.means(), expected_mean, atol=1e-6)

    # weighted covariance reference computed directly
    delta = X - expected_mean
    expected_cov = (w[:, None] * delta).T @ delta / (w.sum() - 1)
    np.testing.assert_allclose(acc.covariance(), expected_cov, atol=1e-6)


def test_canonical_correlation_perfectly_correlated_bands():
    # If target == reference exactly, canonical correlation should be ~1.
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(2000, 2))
    tgt = ref.copy()
    stacked = np.concatenate([ref, tgt], axis=1)
    S = np.cov(stacked, rowvar=False)
    s11, s22, s12 = S[:2, :2], S[2:, 2:], S[:2, 2:]

    rho, A, B = _solve_canonical_correlation(s11, s22, s12)
    assert rho.shape == (2,)
    np.testing.assert_allclose(rho, [1.0, 1.0], atol=1e-6)


def test_canonical_correlation_independent_bands():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(5000, 2))
    tgt = rng.normal(size=(5000, 2))  # independent of ref
    stacked = np.concatenate([ref, tgt], axis=1)
    S = np.cov(stacked, rowvar=False)
    s11, s22, s12 = S[:2, :2], S[2:, 2:], S[:2, 2:]

    rho, A, B = _solve_canonical_correlation(s11, s22, s12)
    assert np.all(rho < 0.15)  # should be near zero for independent data


def test_log_transform_balances_dark_and_bright_invariant_detection(tmp_path):
    # Two brightness groups (dark ~100-150 DN, bright ~1500-3000 DN) with
    # the SAME relative (2%) noise level between reference and target, so
    # both are equally *stable* in relative terms despite very different
    # absolute DN ranges. Raw-DN chi-square is heteroscedastic (absolute
    # noise scales with brightness) and should favor the dark group; log-DN
    # chi-square measures proportional change and should treat them evenly.
    rng = np.random.default_rng(3)
    H, W = 200, 200
    dark_mask = np.zeros((H, W), dtype=bool)
    dark_mask[:100, :] = True
    bright_mask = ~dark_mask

    band0_ref = np.where(dark_mask, 150.0, 3000.0) + rng.normal(0, 5, (H, W))
    band1_ref = np.where(dark_mask, 80.0, 1500.0) + rng.normal(0, 3, (H, W))
    ref_bands_arr = [band0_ref, band1_ref]
    tgt_bands_arr = [np.clip(b * (1.0 + rng.normal(0, 0.02, b.shape)), 1, None) for b in ref_bands_arr]

    ref_path = write_geotiff(str(tmp_path / "ref.tif"), ref_bands_arr, nodata=None)
    tgt_path = write_geotiff(str(tmp_path / "tgt.tif"), tgt_bands_arr, nodata=None)
    search_mask = np.ones((H, W), dtype=bool)
    window = (0, 0, W, H)

    def invariant_fraction_by_group(log_transform):
        fit = irmad.fit_irmad(ref_path, tgt_path, window, window, [1, 2], [1, 2], search_mask,
                               log_transform=log_transform)
        rds, rb = io.open_bands(ref_path, [1, 2])
        tds, tb = io.open_bands(tgt_path, [1, 2])
        tile_ref = io.read_block_flat(rb, 0, 0, W, H)
        tile_tgt = io.read_block_flat(tb, 0, 0, W, H)
        rds = tds = None
        if log_transform:
            tile_ref = irmad._log_transform_tile(tile_ref)
            tile_tgt = irmad._log_transform_tile(tile_tgt)
        chisqr = irmad.mad_chisqr(tile_ref, tile_tgt, fit)
        invariant = spstats.chi2.sf(chisqr, 2) > 0.70
        return invariant[dark_mask.ravel()].mean(), invariant[bright_mask.ravel()].mean()

    dark_raw, bright_raw = invariant_fraction_by_group(log_transform=False)
    dark_log, bright_log = invariant_fraction_by_group(log_transform=True)

    # raw DN: dark group flagged invariant far more often than the equally-
    # stable bright group (heteroscedastic bias)
    assert dark_raw > bright_raw * 5

    # log DN: the two equally-stable groups should be flagged at comparable
    # rates -- neither systematically favored
    assert 0.5 < (dark_log / max(bright_log, 1e-9)) < 2.0


def _make_correlated_ref_tgt(tmp_path, size=60, seed=5):
    """A 2-band ref/target pair with a real (noisy but genuine) linear
    relationship, so IR-MAD has real signal to converge on -- used to check
    that caching the window (instead of re-reading it every iteration)
    doesn't change the fitted result."""
    rng = np.random.default_rng(seed)
    band0 = rng.uniform(200, 3000, (size, size))
    band1 = rng.uniform(100, 1500, (size, size))
    ref_bands_arr = [band0, band1]
    tgt_bands_arr = [np.clip(b * 1.05 + rng.normal(0, 3, b.shape), 1, None) for b in ref_bands_arr]
    ref_path = write_geotiff(str(tmp_path / "ref.tif"), ref_bands_arr, nodata=None)
    tgt_path = write_geotiff(str(tmp_path / "tgt.tif"), tgt_bands_arr, nodata=None)
    search_mask = np.ones((size, size), dtype=bool)
    window = (0, 0, size, size)
    return ref_path, tgt_path, window, search_mask


def test_fit_irmad_return_cache_matches_fit_without_cache(tmp_path):
    # The cached-window code path (return_cache=True) must reach the exact
    # same converged fit as the plain path -- caching only changes *where*
    # pixel data is read from between iterations, never what's computed.
    ref_path, tgt_path, window, search_mask = _make_correlated_ref_tgt(tmp_path)

    fit_plain = irmad.fit_irmad(ref_path, tgt_path, window, window, [1, 2], [1, 2], search_mask)
    fit_cached, cache = irmad.fit_irmad(
        ref_path, tgt_path, window, window, [1, 2], [1, 2], search_mask, return_cache=True,
    )

    np.testing.assert_allclose(fit_cached.rho, fit_plain.rho, atol=1e-10)
    np.testing.assert_allclose(fit_cached.A, fit_plain.A, atol=1e-10)
    np.testing.assert_allclose(fit_cached.B, fit_plain.B, atol=1e-10)
    np.testing.assert_allclose(fit_cached.means1, fit_plain.means1, atol=1e-10)
    np.testing.assert_allclose(fit_cached.means2, fit_plain.means2, atol=1e-10)
    assert fit_cached.n_iterations == fit_plain.n_iterations
    assert fit_cached.converged == fit_plain.converged
    assert cache.device == "cpu"
    assert len(cache.blocks) > 0


def test_fit_irmad_cache_is_reusable_for_a_second_fit_without_disk_access(tmp_path):
    # The whole point of return_cache: a caller can pass the cache back in
    # via `_cache` and get the identical fit without fit_irmad touching disk
    # a second time. Prove this by deleting the source files before the
    # second call -- if it silently re-read from disk, this would raise.
    ref_path, tgt_path, window, search_mask = _make_correlated_ref_tgt(tmp_path)

    fit1, cache = irmad.fit_irmad(ref_path, tgt_path, window, window, [1, 2], [1, 2], search_mask, return_cache=True)

    os.remove(ref_path)
    os.remove(tgt_path)

    fit2 = irmad.fit_irmad(
        ref_path, tgt_path, window, window, [1, 2], [1, 2], search_mask, _cache=cache,
    )
    np.testing.assert_allclose(fit2.rho, fit1.rho, atol=1e-10)
    np.testing.assert_allclose(fit2.A, fit1.A, atol=1e-10)


def test_fit_irmad_default_device_is_cpu_and_matches_explicit_cpu(tmp_path):
    ref_path, tgt_path, window, search_mask = _make_correlated_ref_tgt(tmp_path, seed=9)
    fit_default = irmad.fit_irmad(ref_path, tgt_path, window, window, [1, 2], [1, 2], search_mask)
    fit_explicit_cpu = irmad.fit_irmad(
        ref_path, tgt_path, window, window, [1, 2], [1, 2], search_mask, device="cpu",
    )
    np.testing.assert_allclose(fit_default.rho, fit_explicit_cpu.rho, atol=1e-12)
