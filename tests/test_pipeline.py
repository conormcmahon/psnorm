import json
import os
import shutil

import numpy as np
import pytest
from osgeo import gdal

from psnorm import io, masking, model_io, pipeline

from _raster_utils import write_geotiff


def _make_scene(tmp_path, scene_id, *, sun_elevation, clear_percent, valid_pixels):
    """A 10x10, 3m-pixel, 4-band scene (900 m^2 total) with only
    `valid_pixels` non-nodata pixels (the rest are 0/nodata)."""
    size = (10, 10)
    arr = np.zeros(size, dtype=np.uint16)
    flat = arr.ravel()
    flat[:valid_pixels] = 500
    if valid_pixels > 0:
        # Keep NDWI safely negative (land, not water) wherever there's data.
        pass
    arr = flat.reshape(size)
    nir = arr.copy()
    green = np.where(arr > 0, 100, 0).astype(np.uint16)  # green << nir -> not water
    path = write_geotiff(
        str(tmp_path / f"{scene_id}_3B_AnalyticMS_SR_harmonized_clip.tif"),
        [arr, green, arr, nir], band_descriptions=["blue", "green", "red", "nir"], nodata=0,
    )
    meta_path = tmp_path / f"{scene_id}_metadata.json"
    meta_path.write_text(json.dumps({"properties": {"sun_elevation": sun_elevation, "clear_percent": clear_percent}}))
    return io.Scene(scene_id=scene_id, analytic_path=path, udm2_path=None, metadata_path=str(meta_path), acquired=None)


def test_select_reference_rejects_small_area_scene_and_widens_percentile(tmp_path):
    # Best sun angle but almost no usable area -- must be rejected even
    # though it's the only scene in the initial (10th percentile) pool, so
    # the percentile has to widen until a scene with enough area enters the
    # pool. Loosening stops at the *first* tier with an acceptable image
    # (per spec: "iteratively loosen... until an acceptable image is
    # found"), not the tier that happens to contain the single best scene
    # overall -- large_best only enters the pool at a wider percentile than
    # large_ok, so large_ok (found first) should win.
    small_good_angle = _make_scene(tmp_path, "small_good_angle", sun_elevation=89, clear_percent=99, valid_pixels=3)
    large_ok = _make_scene(tmp_path, "large_ok", sun_elevation=40, clear_percent=60, valid_pixels=100)
    large_best = _make_scene(tmp_path, "large_best", sun_elevation=5, clear_percent=95, valid_pixels=100)

    scenes = [small_good_angle, large_ok, large_best]
    chosen = pipeline.select_reference(
        scenes, zenith_percentile=10.0, zenith_percentile_step=10.0,
        min_reference_area_km2=0.0005,  # 500 m^2; full 900 m^2 scene passes, 3-pixel (27 m^2) scene fails
        use_omnicloudmask=False,
    )
    assert chosen.scene_id == "large_ok"


def test_select_reference_picks_best_clear_fraction_among_scenes_in_the_same_tier(tmp_path):
    # Two large-area scenes with the same (bad) sun angle enter the pool
    # together once the percentile widens far enough -- the one with the
    # higher clear fraction should win between them.
    small_good_angle = _make_scene(tmp_path, "small_good_angle", sun_elevation=89, clear_percent=99, valid_pixels=3)
    large_ok = _make_scene(tmp_path, "large_ok", sun_elevation=5, clear_percent=60, valid_pixels=100)
    large_best = _make_scene(tmp_path, "large_best", sun_elevation=5, clear_percent=95, valid_pixels=100)

    chosen = pipeline.select_reference(
        [small_good_angle, large_ok, large_best], zenith_percentile=10.0, zenith_percentile_step=10.0,
        min_reference_area_km2=0.0005, use_omnicloudmask=False,
    )
    assert chosen.scene_id == "large_best"


def test_select_reference_picks_low_zenith_scene_when_it_has_enough_area(tmp_path):
    good = _make_scene(tmp_path, "good", sun_elevation=89, clear_percent=90, valid_pixels=100)
    other = _make_scene(tmp_path, "other", sun_elevation=10, clear_percent=99, valid_pixels=100)

    chosen = pipeline.select_reference(
        [good, other], zenith_percentile=10.0, min_reference_area_km2=0.0005, use_omnicloudmask=False,
    )
    assert chosen.scene_id == "good"  # lowest zenith already clears the area bar -- no widening needed


def test_select_reference_raises_when_no_scene_ever_has_enough_area(tmp_path):
    tiny_a = _make_scene(tmp_path, "tiny_a", sun_elevation=89, clear_percent=99, valid_pixels=3)
    tiny_b = _make_scene(tmp_path, "tiny_b", sun_elevation=10, clear_percent=50, valid_pixels=3)

    with pytest.raises(ValueError, match="unmasked"):
        pipeline.select_reference(
            [tiny_a, tiny_b], zenith_percentile=10.0, min_reference_area_km2=0.0005, use_omnicloudmask=False,
        )


def test_select_reference_explicit_path_bypasses_area_check(tmp_path):
    tiny = _make_scene(tmp_path, "tiny", sun_elevation=89, clear_percent=99, valid_pixels=3)
    chosen = pipeline.select_reference(
        [tiny], reference_path=tiny.analytic_path, min_reference_area_km2=1000.0, use_omnicloudmask=False,
    )
    assert chosen.scene_id == "tiny"


def _write_varying_scene(tmp_path, scene_id, arrays):
    path = write_geotiff(
        str(tmp_path / f"{scene_id}_3B_AnalyticMS_SR_harmonized_clip.tif"),
        arrays, band_descriptions=["blue", "green", "red", "nir"], nodata=0,
    )
    (tmp_path / f"{scene_id}_metadata.json").write_text(
        json.dumps({"properties": {"sun_elevation": 45, "clear_percent": 95}})
    )
    return path


def _make_correlated_reference_and_target(tmp_path, *, ref_id="20240101_120000_ref", tgt_id="20240102_120000_tgt", size=60, seed=21):
    """A reference scene and a target scene sharing the same grid, with the
    target's bands a noisy linear function of the reference's -- enough real
    (band-correlated) signal for IR-MAD to converge on a non-degenerate fit,
    unlike test_select_reference's constant-fill fixtures (which are fine
    for area/clear-fraction bookkeeping but singular for IR-MAD)."""
    rng = np.random.default_rng(seed)
    ref_arrays = [rng.uniform(lo, hi, (size, size)) for lo, hi in [(200, 3000), (100, 1500), (150, 2000), (300, 3500)]]
    tgt_arrays = [np.clip(a * 1.04 + rng.normal(0, 3, a.shape), 1, None) for a in ref_arrays]
    ref_path = _write_varying_scene(tmp_path, ref_id, [a.astype(np.uint16) for a in ref_arrays])
    tgt_path = _write_varying_scene(tmp_path, tgt_id, [a.astype(np.uint16) for a in tgt_arrays])
    return ref_path, tgt_path


def test_run_pipeline_device_cpu_saves_irmad_fit(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_correlated_reference_and_target(input_dir)

    result = pipeline.run_pipeline(
        str(input_dir), str(output_dir),
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        device="cpu", workers=None,
    )
    assert result.reference_scene_id is not None

    fit_path = pipeline._irmad_fit_path(str(output_dir), "20240102_120000_tgt")
    assert os.path.exists(fit_path)
    fit = model_io.load_irmad_fit(fit_path)
    assert fit.rho.shape == (4,)


def test_run_pipeline_resume_does_not_recompute_saved_irmad_fit(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_correlated_reference_and_target(input_dir)

    pipeline.run_pipeline(
        str(input_dir), str(output_dir),
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        device="cpu", workers=None,
    )
    fit_path = pipeline._irmad_fit_path(str(output_dir), "20240102_120000_tgt")
    mtime_before = os.path.getmtime(fit_path)

    pipeline.run_pipeline(
        str(input_dir), str(output_dir),
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        device="cpu", workers=None, resume="validate",
    )
    assert os.path.getmtime(fit_path) == mtime_before


def test_run_pipeline_resumed_scene_still_reports_identity_fallback_bands(tmp_path):
    # A resumed scene's SceneResult.n_identity_fallback_bands must be
    # re-derived from the saved model, not silently default to 0 -- else
    # summary.md's fallback tally undercounts any run with resumed scenes.
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_correlated_reference_and_target(input_dir)
    kwargs = dict(
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        invariance_frequency_threshold=0.1, min_observations=1,
        device="cpu", workers=None,
        slope_bounds=(0.99, 1.01),  # narrow enough that the ~0.96 true inverse slope always falls outside
    )

    result_fresh = pipeline.run_pipeline(str(input_dir), str(output_dir), **kwargs)
    fresh = next(r for r in result_fresh.scene_results if r.scene_id == "20240102_120000_tgt")
    assert fresh.status == "fitted"
    assert fresh.n_identity_fallback_bands == 4  # all 4 bands rejected by the narrow slope_bounds

    result_resumed = pipeline.run_pipeline(str(input_dir), str(output_dir), resume="validate", **kwargs)
    resumed = next(r for r in result_resumed.scene_results if r.scene_id == "20240102_120000_tgt")
    assert resumed.status == "resumed"
    assert resumed.n_identity_fallback_bands == 4


def test_run_pipeline_rerun_with_different_ncp_threshold_reclassifies_not_refits(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_correlated_reference_and_target(input_dir)
    common_kwargs = dict(
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        device="cpu", workers=None, invariance_frequency_threshold=0.1, min_observations=1,
    )

    pipeline.run_pipeline(str(input_dir), str(output_dir), ncp_threshold=0.70, **common_kwargs)
    fit_path = pipeline._irmad_fit_path(str(output_dir), "20240102_120000_tgt")
    candidate_path = pipeline._candidate_mask_path(str(output_dir), "20240102_120000_tgt")
    stats_path = pipeline._candidate_stats_path(str(output_dir), "20240102_120000_tgt")
    fit_mtime_before = os.path.getmtime(fit_path)
    candidate_mtime_before = os.path.getmtime(candidate_path)

    pipeline.run_pipeline(str(input_dir), str(output_dir), ncp_threshold=0.30, resume="validate", **common_kwargs)

    # the fit itself (expensive to produce) must NOT have been touched --
    # only the cheap classification pass should have rerun.
    assert os.path.getmtime(fit_path) == fit_mtime_before
    # but the candidate mask (and its recorded threshold) must have been
    # refreshed at the new, looser threshold.
    assert os.path.getmtime(candidate_path) > candidate_mtime_before
    stats = model_io.load_json(stats_path)
    assert stats["ncp_threshold"] == 0.30
    assert 0 <= stats["n_invariant"] <= stats["n_evaluated"]


def test_run_pipeline_rerun_with_different_freq_threshold_refits_not_resumes(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_correlated_reference_and_target(input_dir)
    common_kwargs = dict(
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        device="cpu", workers=None, ncp_threshold=0.30, min_observations=1,
    )

    pipeline.run_pipeline(str(input_dir), str(output_dir), invariance_frequency_threshold=0.5, **common_kwargs)
    model_path = pipeline._model_path(str(output_dir), "20240102_120000_tgt")
    model_mtime_before = os.path.getmtime(model_path)
    model_before = model_io.load_model(model_path)

    pipeline.run_pipeline(
        str(input_dir), str(output_dir), invariance_frequency_threshold=0.1, resume="validate", **common_kwargs,
    )

    # a differently-thresholded consensus set must trigger a real refit, not
    # a silent resume of the stale (old-threshold) model.
    assert os.path.getmtime(model_path) > model_mtime_before
    model_after = model_io.load_model(model_path)
    assert model_after.invariance_frequency_threshold == 0.1
    assert model_before.invariance_frequency_threshold == 0.5


def test_run_pipeline_writes_ransac_inlier_rasters_per_band(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_correlated_reference_and_target(input_dir)

    pipeline.run_pipeline(
        str(input_dir), str(output_dir),
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        invariance_frequency_threshold=0.1, min_observations=1,
        device="cpu", workers=None,
    )

    for band_name in ["blue", "green", "red", "nir"]:
        path = pipeline._ransac_inliers_path(str(output_dir), "20240102_120000_tgt", band_name)
        assert os.path.exists(path)
        ds = gdal.Open(path)
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        assert arr.dtype == np.uint8
        assert set(np.unique(arr)).issubset({0, 1})
        assert arr.sum() > 0  # the correlated fixture should have real inliers


def _write_flat_and_steep_dsm(tmp_path, *, scene_size=60, scene_pixel_size=3.0, dsm_pixel_size=1.0):
    """A DSM covering the same footprint as _write_varying_scene's default
    origin/pixel_size, half perfectly flat and half a steep ramp, so
    max_slope_deg=5.0 excludes roughly the ramp half and nothing else."""
    dsm_size = int(scene_size * scene_pixel_size / dsm_pixel_size)
    heights = np.zeros((dsm_size, dsm_size), dtype=np.float32)
    ramp = np.linspace(0.0, 40.0, dsm_size // 2, dtype=np.float32)  # steep: 40m over half the footprint
    heights[:, dsm_size // 2 :] = ramp
    return write_geotiff(
        str(tmp_path / "dsm.tif"), [heights], origin=(0.0, 100.0), pixel_size=dsm_pixel_size,
        nodata=None, dtype=gdal.GDT_Float32,
    )


def test_run_pipeline_dsm_path_writes_lidar_masks_and_search_mask(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_correlated_reference_and_target(input_dir)
    dsm_path = _write_flat_and_steep_dsm(input_dir)

    pipeline.run_pipeline(
        str(input_dir), str(output_dir),
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        invariance_frequency_threshold=0.1, min_observations=1,
        device="cpu", workers=None,
        dsm_path=str(dsm_path), max_slope_deg=5.0,
    )

    for scene_id in ["20240101_120000_ref", "20240102_120000_tgt"]:
        horiz_path = pipeline._lidar_flag_path(str(output_dir), scene_id, "horizontality")
        rough_path = pipeline._lidar_flag_path(str(output_dir), scene_id, "roughness")
        search_path = pipeline._search_mask_path(str(output_dir), scene_id)
        flags_path = pipeline._flags_path(str(output_dir), scene_id)
        assert os.path.exists(horiz_path)
        assert os.path.exists(rough_path)
        assert os.path.exists(search_path)

        horiz_ds = gdal.Open(horiz_path)
        horiz = horiz_ds.GetRasterBand(1).ReadAsArray()
        horiz_ds = None
        assert horiz.any()  # the steep ramp half should be flagged in every scene sharing this footprint
        assert not horiz[:, : horiz.shape[1] // 4].any()  # the flat quarter well away from the ramp is clear

        flags_ds = gdal.Open(flags_path)
        flags = flags_ds.GetRasterBand(1).ReadAsArray()
        flags_ds = None
        assert ((flags & masking.EXCLUDE_LIDAR_SLOPE) != 0).any()

        # raw flags and the eroded/dilated search mask are saved separately,
        # per the "new separate file" design -- both exist and both reflect
        # the slope exclusion, but they need not be pixel-identical.
        search_ds = gdal.Open(search_path)
        search = search_ds.GetRasterBand(1).ReadAsArray()
        search_ds = None
        assert ((search & masking.EXCLUDE_LIDAR_SLOPE) != 0).any()


def test_run_pipeline_dsm_lidar_masks_cover_target_scenes_beyond_reference_extent(tmp_path):
    # Regression test for the union-extent bug: the DSM window used to
    # build the shared slope/roughness cache must cover every scene's
    # footprint, not just the reference's -- a target scene extending
    # beyond the reference would otherwise get zero LiDAR flagging in the
    # part outside the reference, even where the DSM has real data there.
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    size, pixel_size = 60, 3.0
    ref_id, tgt_id = "20240101_120000_ref", "20240102_120000_tgt"

    rng = np.random.default_rng(7)
    ref_arrays = [rng.uniform(lo, hi, (size, size)) for lo, hi in [(200, 3000), (100, 1500), (150, 2000), (300, 3500)]]
    tgt_arrays = [np.clip(a * 1.04 + rng.normal(0, 3, a.shape), 1, None) for a in ref_arrays]

    ref_path = write_geotiff(
        str(input_dir / f"{ref_id}_3B_AnalyticMS_SR_harmonized_clip.tif"),
        [a.astype(np.uint16) for a in ref_arrays], band_descriptions=["blue", "green", "red", "nir"], nodata=0,
        origin=(0.0, 100.0), pixel_size=pixel_size,
    )
    # Shifted 90m (30 px) north: half of the target's footprint (its own
    # northern half) falls entirely outside the reference's extent.
    tgt_path = write_geotiff(
        str(input_dir / f"{tgt_id}_3B_AnalyticMS_SR_harmonized_clip.tif"),
        [a.astype(np.uint16) for a in tgt_arrays], band_descriptions=["blue", "green", "red", "nir"], nodata=0,
        origin=(0.0, 100.0 + size * pixel_size / 2), pixel_size=pixel_size,
    )
    for scene_id in (ref_id, tgt_id):
        (input_dir / f"{scene_id}_metadata.json").write_text(
            json.dumps({"properties": {"sun_elevation": 45, "clear_percent": 95}})
        )

    # DSM covering the full union extent (both scenes), all steep (>5 deg)
    # so every LiDAR-covered pixel in either scene gets flagged -- isolates
    # "was this area ever read from the DSM at all" from slope math details.
    dsm_size = int(size * pixel_size / 1.0 + size * pixel_size / 2 / 1.0) + 10
    ramp = np.tile(np.linspace(0.0, 200.0, dsm_size, dtype=np.float32), (dsm_size, 1))
    dsm_path = write_geotiff(
        str(input_dir / "dsm.tif"), [ramp], origin=(-5.0, 100.0 + size * pixel_size / 2 + 5.0), pixel_size=1.0,
        nodata=None, dtype=gdal.GDT_Float32,
    )

    pipeline.run_pipeline(
        str(input_dir), str(output_dir),
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        invariance_frequency_threshold=0.1, min_observations=1,
        device="cpu", workers=None, reference_path=ref_path,
        dsm_path=str(dsm_path), max_slope_deg=5.0,
    )

    horiz_path = pipeline._lidar_flag_path(str(output_dir), tgt_id, "horizontality")
    ds = gdal.Open(horiz_path)
    horiz = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    # Target's own far-north rows (well outside the reference's footprint,
    # and far enough from the reference/target boundary that the small
    # roughness-window buffer -- a few meters -- can't accidentally reach
    # this far) must still show LiDAR flagging -- before the fix this was
    # always zero there regardless of the DSM's content.
    assert horiz[:10, :].any()


def test_run_pipeline_dsm_path_none_skips_lidar_masks_entirely(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_correlated_reference_and_target(input_dir)

    pipeline.run_pipeline(
        str(input_dir), str(output_dir),
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        device="cpu", workers=None,
    )
    assert not os.path.exists(pipeline._lidar_flag_path(str(output_dir), "20240101_120000_ref", "horizontality"))
    flags_path = pipeline._flags_path(str(output_dir), "20240101_120000_ref")
    flags_ds = gdal.Open(flags_path)
    flags = flags_ds.GetRasterBand(1).ReadAsArray()
    flags_ds = None
    assert not ((flags & masking.EXCLUDE_LIDAR_SLOPE) != 0).any()


def test_run_pipeline_device_gpu_without_gpu_raises(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_correlated_reference_and_target(input_dir)

    with pytest.raises(RuntimeError, match="no usable GPU"):
        pipeline.run_pipeline(
            str(input_dir), str(output_dir),
            use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
            min_reference_area_km2=0.001, device="gpu", workers=None,
        )


def test_run_pipeline_reads_registered_scenes_from_a_preferred_imagery_manifest(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    ref_path, tgt_path = _make_correlated_reference_and_target(input_dir)
    # The target's "registered" version: same grid and file name, elsewhere.
    registered_dir = tmp_path / "affine_transform"
    registered_dir.mkdir()
    registered_path = str(registered_dir / os.path.basename(tgt_path))
    shutil.copyfile(tgt_path, registered_path)
    manifest = tmp_path / "preferred_imagery.csv"
    manifest.write_text(
        "scene_id,preferred_source,preferred_analytic_path,raw_analytic_path\n"
        f"20240101_120000_ref,raw,{ref_path},{ref_path}\n"
        f"20240102_120000_tgt,registered,{registered_path},{tgt_path}\n"
    )
    applied = []
    real_apply = pipeline.apply.apply_model
    monkeypatch.setattr(pipeline.apply, "apply_model",
                        lambda path, *a, **kw: applied.append(path) or real_apply(path, *a, **kw))

    result = pipeline.run_pipeline(
        str(input_dir), str(tmp_path / "output"), reference_path=ref_path,
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        device="cpu", workers=None, min_observations=1, preferred_imagery_manifest=str(manifest),
    )
    assert result.reference_scene_id == "20240101_120000_ref"
    assert registered_path in applied and tgt_path not in applied


def test_phase_c_fits_only_consensus_pixels_clear_in_that_target(tmp_path, monkeypatch):
    # The consensus set is study-wide, so a pixel can be in it while being
    # cloudy in one particular target. That target's fit must skip it.
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _make_correlated_reference_and_target(input_dir, tgt_id="20240102_120000_cloudy")
    _make_correlated_reference_and_target(input_dir, tgt_id="20240103_120000_clear", seed=21)
    clear_band = np.ones((60, 60), dtype=np.uint8)
    clear_band[10:30, 10:30] = 0  # a cloud in the first target only
    write_geotiff(str(input_dir / "20240102_120000_cloudy_3B_udm2_clip.tif"),
                  [clear_band] + [np.zeros((60, 60), dtype=np.uint8)] * 7)

    fitted_masks = {}
    real_fit = pipeline.normalize.fit_regression_from_mask

    def recording_fit(reference_path, target_path, reference_id, target_id, ref_window, tgt_window,
                      band_indices_ref, band_indices_tgt, band_names, consensus_mask, **kw):
        fitted_masks[target_id] = consensus_mask.copy()
        return real_fit(reference_path, target_path, reference_id, target_id, ref_window, tgt_window,
                        band_indices_ref, band_indices_tgt, band_names, consensus_mask, **kw)

    monkeypatch.setattr(pipeline.normalize, "fit_regression_from_mask", recording_fit)
    pipeline.run_pipeline(
        str(input_dir), str(tmp_path / "output"),
        reference_path=str(input_dir / "20240101_120000_ref_3B_AnalyticMS_SR_harmonized_clip.tif"),
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        invariance_frequency_threshold=0.1, min_observations=1, device="cpu", workers=None,
    )
    # Inside the cloud (away from the search mask's 1-pixel erode/dilate
    # margin): consensus pixels there are fitted for the clear target only.
    assert fitted_masks["20240103_120000_clear"][12:28, 12:28].any()
    assert not fitted_masks["20240102_120000_cloudy"][12:28, 12:28].any()
    assert fitted_masks["20240102_120000_cloudy"].any()


def test_saved_model_with_different_ransac_settings_is_refit(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    _make_correlated_reference_and_target(input_dir)
    kwargs = dict(use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
                  min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
                  invariance_frequency_threshold=0.1, min_observations=1, device="cpu", workers=None)
    pipeline.run_pipeline(str(input_dir), str(output_dir), **kwargs)
    model_path = pipeline._model_path(str(output_dir), "20240102_120000_tgt")
    assert model_io.load_model(model_path).fit_settings["outlier_abs_threshold_dn"] == 200.0

    same = pipeline.run_pipeline(str(input_dir), str(output_dir), resume="validate", **kwargs)
    assert {r.scene_id: r.status for r in same.scene_results}["20240102_120000_tgt"] == "resumed"
    changed = pipeline.run_pipeline(str(input_dir), str(output_dir), resume="validate",
                                    outlier_abs_threshold_dn=150.0, **kwargs)
    assert {r.scene_id: r.status for r in changed.scene_results}["20240102_120000_tgt"] == "fitted"
