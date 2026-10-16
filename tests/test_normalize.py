import math

import numpy as np
import pytest
from osgeo import gdal

from psnorm import normalize
from psnorm.normalize import _RegressionAccumulator, fit_orthogonal_regression

from _raster_utils import write_geotiff


def test_orthogonal_regression_recovers_known_line_no_noise():
    x = np.linspace(0, 100, 200)
    true_slope, true_intercept = 1.5, 10.0
    y = true_intercept + true_slope * x

    slope, intercept, r2 = fit_orthogonal_regression(x, y)
    assert slope == pytest.approx(true_slope, abs=1e-6)
    assert intercept == pytest.approx(true_intercept, abs=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_orthogonal_regression_recovers_known_line_with_noise():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1000, 5000)
    true_slope, true_intercept = 0.8, -25.0
    y = true_intercept + true_slope * x + rng.normal(0, 5, size=x.shape)

    slope, intercept, r2 = fit_orthogonal_regression(x, y)
    assert slope == pytest.approx(true_slope, abs=0.05)
    assert intercept == pytest.approx(true_intercept, abs=5.0)
    assert r2 > 0.99


def test_regression_accumulator_matches_batch_computation():
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 500, 1000)
    y = 2.0 + 0.5 * x + rng.normal(0, 2, size=x.shape)

    batch_slope, batch_intercept, batch_r2 = fit_orthogonal_regression(x, y)

    acc = _RegressionAccumulator()
    acc.update(x[:400], y[:400])
    acc.update(x[400:], y[400:])
    slope, intercept, r2, n = acc.finalize()

    assert n == 1000
    assert slope == pytest.approx(batch_slope, abs=1e-8)
    assert intercept == pytest.approx(batch_intercept, abs=1e-8)
    assert r2 == pytest.approx(batch_r2, abs=1e-8)


def test_regression_accumulator_too_few_points_raises():
    acc = _RegressionAccumulator()
    acc.update(np.array([1.0]), np.array([2.0]))
    with pytest.raises(ValueError):
        acc.finalize()


def test_ransac_abs_inliers_uses_an_absolute_dn_threshold():
    x = np.array([100.0, 2000.0, 300.0, 2000.0])
    slope, intercept = 1.0, 0.0
    # The same 150 DN deviation counts as an inlier for a dark and a bright
    # target alike; 250 DN does not, at either level.
    y = np.array([250.0, 2150.0, 550.0, 2250.0])
    inliers = normalize._ransac_abs_inliers(x, y, slope, intercept, 200.0)
    np.testing.assert_array_equal(inliers, [True, True, False, False])


def test_ransac_defaults():
    assert normalize.DEFAULT_OUTLIER_ABS_THRESHOLD_DN == 200.0
    assert normalize.DEFAULT_SLOPE_BOUNDS == pytest.approx((1 / 1.5, 1.5))
    assert normalize.DEFAULT_RANSAC_MAX_ITERATIONS == 10_000
    assert normalize.DEFAULT_RANSAC_SCORE_SUBSAMPLE == 20_000


def test_ransac_subsampled_scoring_still_classifies_every_point():
    rng = np.random.default_rng(3)
    n = 60_000
    x = rng.uniform(200, 4000, n)
    y = 30.0 + 1.05 * x + rng.normal(0, 40.0, n)
    bad = rng.random(n) < 0.25
    y[bad] += rng.choice([-1, 1], bad.sum()) * rng.uniform(400, 2000, bad.sum())

    _, _, n_sub, mask_sub = normalize._ransac_orthogonal_regression(
        x, y, 200.0, max_iterations=500, slope_bounds=normalize.DEFAULT_SLOPE_BOUNDS, score_subsample=5_000)
    _, _, n_all, mask_all = normalize._ransac_orthogonal_regression(
        x, y, 200.0, max_iterations=500, slope_bounds=normalize.DEFAULT_SLOPE_BOUNDS, score_subsample=None)

    # Outliers among the points outside the scoring subset are still removed.
    assert mask_sub.shape == (n,)
    assert not mask_sub[bad].any()
    assert mask_sub[~bad].mean() > 0.99
    # Same answer as scoring on every point, to within a handful of
    # borderline points.
    assert abs(n_sub - n_all) < 0.001 * n
    slope, intercept, _ = fit_orthogonal_regression(x[mask_sub], y[mask_sub])
    assert slope == pytest.approx(1.05, abs=0.005)
    assert intercept == pytest.approx(30.0, abs=15.0)


def test_ransac_score_subsample_has_no_effect_below_its_size():
    rng = np.random.default_rng(4)
    x = rng.uniform(200, 4000, 3_000)
    y = 30.0 + 1.05 * x + rng.normal(0, 40.0, x.size)
    y[:300] += 900.0
    a = normalize._ransac_orthogonal_regression(x, y, 200.0, max_iterations=200, score_subsample=5_000)
    b = normalize._ransac_orthogonal_regression(x, y, 200.0, max_iterations=200, score_subsample=None)
    np.testing.assert_array_equal(a[3], b[3])


def test_ransac_orthogonal_regression_removes_absolute_outliers_at_the_default_threshold():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(500, 4000, n)
    y = 30.0 + 1.05 * x + rng.normal(0, 20.0, n)
    idx = rng.choice(n, size=10, replace=False)
    y_contaminated = y.copy()
    y_contaminated[idx] += 400.0  # e.g. haze: 0.04 reflectance too bright
    _, _, n_excluded, inlier_mask = normalize._ransac_orthogonal_regression(
        x, y_contaminated, normalize.DEFAULT_OUTLIER_ABS_THRESHOLD_DN, slope_bounds=normalize.DEFAULT_SLOPE_BOUNDS)
    assert n_excluded == 10
    assert not inlier_mask[idx].any()


def test_ransac_orthogonal_regression_removes_relative_outliers():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(100, 1000, n)
    true_slope, true_intercept = 1.2, 50.0
    y = true_intercept + true_slope * x + rng.normal(0, 1.0, n)  # tight, well-correlated

    # 10 points whose *reference* value is 50% off the true relationship --
    # far outside any of the thresholds this feature is meant to test (2-15%)
    idx = rng.choice(n, size=10, replace=False)
    y_contaminated = y.copy()
    y_contaminated[idx] *= 1.5

    x_clean, y_clean, n_excluded, inlier_mask = normalize._ransac_orthogonal_regression(x, y_contaminated, 20.0)
    assert n_excluded == 10
    assert x_clean.size == n - 10
    assert inlier_mask.shape == (n,)
    assert inlier_mask.sum() == n - 10
    np.testing.assert_array_equal(x[inlier_mask], x_clean)

    slope, intercept, r2 = fit_orthogonal_regression(x_clean, y_clean)
    assert slope == pytest.approx(true_slope, abs=0.05)
    assert intercept == pytest.approx(true_intercept, abs=10.0)
    assert r2 > 0.99


def test_ransac_orthogonal_regression_no_op_when_too_few_points():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    x2, y2, n, inlier_mask = normalize._ransac_orthogonal_regression(x, y, 20.0)
    np.testing.assert_array_equal(x2, x)
    np.testing.assert_array_equal(y2, y)
    assert n == 0
    np.testing.assert_array_equal(inlier_mask, [True, True])


def test_ransac_orthogonal_regression_slope_bounds_rejects_implausible_candidates():
    # A minimal-sample candidate whose own slope falls outside slope_bounds
    # must never be scored/eligible to win, regardless of how many points it
    # would otherwise explain -- construct a case where the *only* points in
    # the data are consistent with an implausible relationship, so any
    # candidate that would normally win is filtered out and the search must
    # fall back to "no candidate found" (same as too-few-points).
    x = np.linspace(100, 1000, 50)
    y = 20.0 + 0.1 * x + np.random.default_rng(3).normal(0, 1.0, 50)  # slope 0.1, far outside (0.8, 1.2)

    x_trimmed, y_trimmed, n_excluded, inlier_mask = normalize._ransac_orthogonal_regression(
        x, y, 20.0, slope_bounds=(0.8, 1.2),
    )
    # no plausible candidate exists anywhere in this data -- falls back to
    # the untrimmed input, deferring rejection to the caller's own
    # min_fit_pixels/slope_bounds post-hoc check (see fit_regression_from_mask).
    assert n_excluded == 0
    np.testing.assert_array_equal(x_trimmed, x)
    np.testing.assert_array_equal(inlier_mask, np.ones(50, dtype=bool))

    # the *same* data without slope_bounds finds and accepts this
    # (otherwise-perfectly-good, just implausible) relationship instead.
    x_unbounded, y_unbounded, n_excluded_unbounded, mask_unbounded = normalize._ransac_orthogonal_regression(
        x, y, 20.0,
    )
    slope, _intercept, _r2 = fit_orthogonal_regression(x_unbounded, y_unbounded)
    assert slope == pytest.approx(0.1, abs=0.05)
    assert mask_unbounded.sum() >= 45  # the clean (if implausible) relationship is found and mostly kept


def _make_ref_tgt_rasters(tmp_path, size, true_slope, true_intercept, outlier_indices, outlier_multiplier,
                          base=100.0):
    # true_slope/true_intercept describe tgt = true_intercept + true_slope*ref
    # (the physical drift being simulated); fit_regression_from_mask fits
    # the *inverse* relationship (reference = a + b*target, target=x), so
    # callers should expect the fitted slope/intercept to be
    # 1/true_slope, -true_intercept/true_slope -- not true_slope/true_intercept
    # directly. `outlier_indices` are contaminated in the *reference* array
    # (multiplicatively, by `outlier_multiplier`) after tgt is derived from
    # the clean reference, so the relative deviation from the fit's own
    # prediction is large and predictable regardless of the fitted slope.
    rng = np.random.default_rng(1)
    n = size * size
    ref = (np.arange(n).reshape(size, size).astype(np.float64) * 10.0 + base)
    noise = rng.normal(0, 2.0, (size, size))
    tgt = true_intercept + true_slope * ref + noise

    flat_ref = ref.ravel().copy()
    for idx in outlier_indices:
        flat_ref[idx] *= outlier_multiplier
    ref_written = flat_ref.reshape(size, size)

    ref_path = write_geotiff(str(tmp_path / "ref.tif"), [ref_written], band_descriptions=["nir"],
                              nodata=None, dtype=gdal.GDT_Float32)
    tgt_path = write_geotiff(str(tmp_path / "tgt.tif"), [tgt], band_descriptions=["nir"],
                              nodata=None, dtype=gdal.GDT_Float32)
    return ref_path, tgt_path


def test_fit_regression_from_mask_finds_plausible_minority_despite_majority_contamination(tmp_path):
    # 55% of consensus pixels are contaminated with an implausible relation-
    # ship (slope far outside (0.8, 1.2)); the genuine, plausible-slope
    # relationship is only the 45% minority. Without slope_bounds enforced
    # during RANSAC's search, the majority contaminated cluster would win by
    # sheer inlier count and get rejected post-hoc as identity_fallback --
    # with it, the search should find the plausible minority instead.
    size = 20  # 400 consensus pixels
    true_slope, true_intercept = 1.05, 5.0
    outlier_indices = list(range(220))  # 220/400 = 55% contaminated
    # base=1000: doubling a reference value of at least 1000 DN puts every
    # contaminated point well outside the default 200 DN tolerance (a
    # doubled 100 DN pixel would sit within it).
    ref_path, tgt_path = _make_ref_tgt_rasters(tmp_path, size, true_slope, true_intercept, outlier_indices, 2.0,
                                               base=1000.0)
    consensus_mask = np.ones((size, size), dtype=bool)

    model, inlier_masks = normalize.fit_regression_from_mask(
        ref_path, tgt_path, "ref_scene", "tgt_scene",
        (0, 0, size, size), (0, 0, size, size), [1], [1], ["nir"], consensus_mask,
        invariance_frequency_threshold=0.5, min_observations=1, min_fit_pixels=2,
        return_inlier_masks=True,
    )
    band = model.band("nir")
    assert not band.identity_fallback
    assert band.slope == pytest.approx(1.0 / true_slope, abs=0.05)

    assert "nir" in inlier_masks
    inlier_raster = inlier_masks["nir"]
    assert inlier_raster.shape == (size, size)
    assert inlier_raster.dtype == bool
    # the accepted inliers should land (almost) entirely on the clean
    # (non-contaminated) pixels -- indices >= 220 in row-major order.
    contaminated_mask = np.zeros(size * size, dtype=bool)
    contaminated_mask[outlier_indices] = True
    contaminated_mask = contaminated_mask.reshape(size, size)
    assert inlier_raster[~contaminated_mask].sum() >= inlier_raster.sum() - 10


def test_fit_regression_from_mask_excludes_outliers_by_default(tmp_path):
    size = 10  # 100 consensus pixels
    true_slope, true_intercept = 1.1, -20.0
    outlier_indices = [3, 20, 45, 67, 88]  # 5% contamination
    # 50% multiplicative distortion in the reference -- far outside the
    # default 5% relative threshold
    ref_path, tgt_path = _make_ref_tgt_rasters(tmp_path, size, true_slope, true_intercept, outlier_indices, 1.5)
    consensus_mask = np.ones((size, size), dtype=bool)

    # min_fit_pixels bypassed here (this test's 100-pixel grid is below the
    # pipeline's real-world default floor of 500) -- fallback behavior has
    # its own dedicated tests below.
    model = normalize.fit_regression_from_mask(
        ref_path, tgt_path, "ref_scene", "tgt_scene",
        (0, 0, size, size), (0, 0, size, size), [1], [1], ["nir"], consensus_mask,
        invariance_frequency_threshold=0.5, min_observations=1, min_fit_pixels=2, outlier_abs_threshold_dn=20.0,
    )
    band = model.band("nir")
    assert band.n_outliers_excluded == len(outlier_indices)
    assert band.n_invariant_pixels == size * size - len(outlier_indices)
    assert band.slope == pytest.approx(1.0 / true_slope, abs=0.02)
    assert band.intercept == pytest.approx(-true_intercept / true_slope, abs=2.0)
    assert band.rmse_after < band.rmse_before
    assert band.r2 > 0.99
    assert not band.identity_fallback
    # model-level consensus count is unaffected by per-band outlier removal
    assert model.n_consensus_pixels == size * size


def test_fit_regression_from_mask_outlier_removal_can_be_disabled(tmp_path):
    size = 10
    true_slope, true_intercept = 1.1, -20.0
    outlier_indices = [3, 20, 45, 67, 88]
    ref_path, tgt_path = _make_ref_tgt_rasters(tmp_path, size, true_slope, true_intercept, outlier_indices, 1.5)
    consensus_mask = np.ones((size, size), dtype=bool)

    model_trimmed = normalize.fit_regression_from_mask(
        ref_path, tgt_path, "ref_scene", "tgt_scene",
        (0, 0, size, size), (0, 0, size, size), [1], [1], ["nir"], consensus_mask,
        invariance_frequency_threshold=0.5, min_observations=1, min_fit_pixels=2, outlier_abs_threshold_dn=20.0,
    )
    model_untrimmed = normalize.fit_regression_from_mask(
        ref_path, tgt_path, "ref_scene", "tgt_scene",
        (0, 0, size, size), (0, 0, size, size), [1], [1], ["nir"], consensus_mask,
        invariance_frequency_threshold=0.5, min_observations=1,
        outlier_abs_threshold_dn=None, min_fit_pixels=2,
    )
    band = model_untrimmed.band("nir")
    assert band.n_outliers_excluded == 0
    assert band.n_invariant_pixels == size * size
    # left in, the outliers should pull r2 down noticeably vs the trimmed fit
    assert band.r2 < model_trimmed.band("nir").r2


def test_fit_regression_from_mask_falls_back_to_identity_when_too_few_pixels(tmp_path):
    size = 10  # 100 consensus pixels, well below a realistic min_fit_pixels floor
    true_slope, true_intercept = 1.1, -20.0
    ref_path, tgt_path = _make_ref_tgt_rasters(tmp_path, size, true_slope, true_intercept, [], 1.0)
    consensus_mask = np.ones((size, size), dtype=bool)

    model = normalize.fit_regression_from_mask(
        ref_path, tgt_path, "ref_scene", "tgt_scene",
        (0, 0, size, size), (0, 0, size, size), [1], [1], ["nir"], consensus_mask,
        invariance_frequency_threshold=0.5, min_observations=1,
        outlier_abs_threshold_dn=None, min_fit_pixels=500,
    )
    band = model.band("nir")
    assert band.identity_fallback
    assert "min_fit_pixels" in band.fallback_reason
    assert band.slope == 1.0 and band.intercept == 0.0
    # the rejected fit is still recorded for inspection
    assert band.raw_slope == pytest.approx(1.0 / true_slope, abs=0.02)
    assert band.raw_intercept is not None
    # r2/rmse still describe the underlying (rejected) fit's own quality
    assert band.r2 > 0.99


def test_fit_regression_from_mask_falls_back_to_identity_when_slope_out_of_bounds(tmp_path):
    size = 100  # 10000 consensus pixels, comfortably above any min_fit_pixels floor
    true_slope, true_intercept = 3.0, 0.0  # inverse fit slope = 1/3 = 0.33, outside [0.8, 1.2]
    ref_path, tgt_path = _make_ref_tgt_rasters(tmp_path, size, true_slope, true_intercept, [], 1.0)
    consensus_mask = np.ones((size, size), dtype=bool)

    model = normalize.fit_regression_from_mask(
        ref_path, tgt_path, "ref_scene", "tgt_scene",
        (0, 0, size, size), (0, 0, size, size), [1], [1], ["nir"], consensus_mask,
        invariance_frequency_threshold=0.5, min_observations=1,
        outlier_abs_threshold_dn=None, min_fit_pixels=500, slope_bounds=(0.8, 1.2),
    )
    band = model.band("nir")
    assert band.identity_fallback
    assert "outside plausible bounds" in band.fallback_reason
    assert band.slope == 1.0 and band.intercept == 0.0
    assert band.raw_slope == pytest.approx(1.0 / true_slope, rel=0.05)


def test_fit_regression_from_mask_slope_bounds_can_be_disabled(tmp_path):
    size = 100
    true_slope, true_intercept = 3.0, 0.0
    ref_path, tgt_path = _make_ref_tgt_rasters(tmp_path, size, true_slope, true_intercept, [], 1.0)
    consensus_mask = np.ones((size, size), dtype=bool)

    model = normalize.fit_regression_from_mask(
        ref_path, tgt_path, "ref_scene", "tgt_scene",
        (0, 0, size, size), (0, 0, size, size), [1], [1], ["nir"], consensus_mask,
        invariance_frequency_threshold=0.5, min_observations=1,
        outlier_abs_threshold_dn=None, min_fit_pixels=500, slope_bounds=None,
    )
    band = model.band("nir")
    assert not band.identity_fallback
    assert band.raw_slope is None
    assert band.slope == pytest.approx(1.0 / true_slope, rel=0.05)


def test_fit_regression_from_mask_no_fallback_for_well_supported_sane_fit(tmp_path):
    size = 100  # 10000 consensus pixels
    true_slope, true_intercept = 1.1, -20.0
    ref_path, tgt_path = _make_ref_tgt_rasters(tmp_path, size, true_slope, true_intercept, [], 1.0)
    consensus_mask = np.ones((size, size), dtype=bool)

    model = normalize.fit_regression_from_mask(
        ref_path, tgt_path, "ref_scene", "tgt_scene",
        (0, 0, size, size), (0, 0, size, size), [1], [1], ["nir"], consensus_mask,
        invariance_frequency_threshold=0.5, min_observations=1,
    )
    band = model.band("nir")
    assert not band.identity_fallback
    assert band.fallback_reason is None
    assert band.raw_slope is None and band.raw_intercept is None
    assert band.slope == pytest.approx(1.0 / true_slope, abs=0.02)


def test_compute_spectral_coverage_reports_variance_and_range(tmp_path):
    band = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
    path = write_geotiff(
        str(tmp_path / "a.tif"), [band], band_descriptions=["nir"], nodata=None, dtype=gdal.GDT_Float32,
    )
    # a narrow slice in the middle of the value range -- invariant targets
    # clustered here should show low variance and partial spectral coverage
    invariant_mask = (band >= 45) & (band <= 55)

    results = normalize.compute_spectral_coverage(path, (0, 0, 10, 10), [1], ["nir"], invariant_mask)
    assert len(results) == 1
    r = results[0]

    target_vals = band[invariant_mask]
    whole_vals = band.ravel()
    assert r.n_target_pixels == target_vals.size
    assert r.target_variance == pytest.approx(float(np.var(target_vals, ddof=1)))
    assert r.whole_image_variance == pytest.approx(float(np.var(whole_vals, ddof=1)))
    assert r.variance_ratio == pytest.approx(r.target_variance / r.whole_image_variance)
    p10, p90 = np.percentile(whole_vals, [10, 90])
    assert r.whole_image_p10 == pytest.approx(p10)
    assert r.whole_image_p90 == pytest.approx(p90)
    assert r.target_min == pytest.approx(target_vals.min())
    assert r.target_max == pytest.approx(target_vals.max())
    # target range [45,55] sits well inside [p10,p90] but doesn't span it
    assert 0 < r.coverage_fraction < 1
    # narrow uniform band has much lower variance than the whole 1..100 range
    assert r.variance_ratio < 0.1


def _make_detectable_ref_tgt(tmp_path, size=60, seed=11):
    rng = np.random.default_rng(seed)
    band0 = rng.uniform(200, 3000, (size, size))
    band1 = rng.uniform(100, 1500, (size, size))
    ref_bands_arr = [band0, band1]
    tgt_bands_arr = [np.clip(b * 1.03 + rng.normal(0, 3, b.shape), 1, None) for b in ref_bands_arr]
    ref_path = write_geotiff(str(tmp_path / "ref.tif"), ref_bands_arr, nodata=None)
    tgt_path = write_geotiff(str(tmp_path / "tgt.tif"), tgt_bands_arr, nodata=None)
    search_mask = np.ones((size, size), dtype=bool)
    window = (0, 0, size, size)
    return ref_path, tgt_path, window, search_mask


def test_detect_invariant_candidates_includes_the_fitted_irmad_model(tmp_path):
    ref_path, tgt_path, window, search_mask = _make_detectable_ref_tgt(tmp_path)
    result = normalize.detect_invariant_candidates(
        ref_path, tgt_path, "ref_scene", "tgt_scene", window, window, [1, 2], [1, 2], ["blue", "green"],
        search_mask, ncp_threshold=0.70,
    )
    assert result.irmad_fit is not None
    assert result.irmad_fit.converged == result.irmad_converged
    np.testing.assert_array_equal(result.irmad_fit.rho, np.asarray(result.irmad_rho))


def test_reclassify_invariant_pixels_matches_detect_at_same_threshold(tmp_path):
    # Reclassifying from a *saved* fit at the same ncp_threshold the fit was
    # originally computed at should reproduce the exact same mask -- this is
    # the guarantee the sweep script depends on (only the threshold changes,
    # not the underlying fit).
    ref_path, tgt_path, window, search_mask = _make_detectable_ref_tgt(tmp_path)
    result = normalize.detect_invariant_candidates(
        ref_path, tgt_path, "ref_scene", "tgt_scene", window, window, [1, 2], [1, 2], ["blue", "green"],
        search_mask, ncp_threshold=0.70,
    )

    mask2, n_eval2, n_inv2 = normalize.reclassify_invariant_pixels(
        ref_path, tgt_path, window, window, [1, 2], [1, 2], search_mask, result.irmad_fit,
        ncp_threshold=0.70,
    )

    np.testing.assert_array_equal(mask2, result.invariant_mask)
    assert n_eval2 == result.n_evaluated
    assert n_inv2 == result.n_invariant


def test_reclassify_invariant_pixels_lower_threshold_yields_more_or_equal_candidates(tmp_path):
    ref_path, tgt_path, window, search_mask = _make_detectable_ref_tgt(tmp_path)
    result = normalize.detect_invariant_candidates(
        ref_path, tgt_path, "ref_scene", "tgt_scene", window, window, [1, 2], [1, 2], ["blue", "green"],
        search_mask, ncp_threshold=0.70,
    )

    _mask_strict, _n_eval_strict, n_inv_strict = normalize.reclassify_invariant_pixels(
        ref_path, tgt_path, window, window, [1, 2], [1, 2], search_mask, result.irmad_fit, ncp_threshold=0.90,
    )
    _mask_loose, _n_eval_loose, n_inv_loose = normalize.reclassify_invariant_pixels(
        ref_path, tgt_path, window, window, [1, 2], [1, 2], search_mask, result.irmad_fit, ncp_threshold=0.30,
    )
    assert n_inv_loose >= n_inv_strict


def test_compute_spectral_coverage_handles_no_target_pixels(tmp_path):
    band = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
    path = write_geotiff(
        str(tmp_path / "a.tif"), [band], band_descriptions=["nir"], nodata=None, dtype=gdal.GDT_Float32,
    )
    empty_mask = np.zeros((10, 10), dtype=bool)

    results = normalize.compute_spectral_coverage(path, (0, 0, 10, 10), [1], ["nir"], empty_mask)
    r = results[0]
    assert r.n_target_pixels == 0
    assert math.isnan(r.target_variance)
    assert math.isnan(r.coverage_fraction)
