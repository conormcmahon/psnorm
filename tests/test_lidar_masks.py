import os

import numpy as np
import pytest
from osgeo import gdal, osr

from psnorm import io, lidar_masks, masking

from _raster_utils import write_geotiff

gdal.UseExceptions()


def _write_dsm(path, heights, *, origin=(0.0, 100.0), pixel_size=1.5, nodata=-9999.0, epsg=32618):
    dataset = write_geotiff(
        path, [heights.astype(np.float32)], origin=origin, pixel_size=pixel_size,
        nodata=nodata, dtype=gdal.GDT_Float32,
    )
    if epsg != 32618:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        ds = gdal.Open(path, gdal.GA_Update)
        ds.SetProjection(srs.ExportToWkt())
        ds = None
    return dataset


def test_height_unit_factor_meters_and_feet():
    assert lidar_masks.height_unit_factor("m") == 1.0
    assert lidar_masks.height_unit_factor("meters") == 1.0
    assert lidar_masks.height_unit_factor("ft") == pytest.approx(0.3048, abs=1e-4)
    assert lidar_masks.height_unit_factor("feet") == pytest.approx(0.3048, abs=1e-4)
    with pytest.raises(ValueError):
        lidar_masks.height_unit_factor("cubits")


def test_dsm_pixel_size_m_converts_feet_crs_to_meters(tmp_path):
    # EPSG:2263 == NAD83 / New York Long Island (ftUS), the same family of
    # CRS as the real test DSM -- pixel size stored in feet must come back
    # in meters.
    heights = np.zeros((4, 4), dtype=np.float32)
    path = _write_dsm(str(tmp_path / "dsm.tif"), heights, pixel_size=5.0, epsg=2263)
    dataset = gdal.Open(path)
    px_m, py_m = lidar_masks.dsm_pixel_size_m(dataset)
    dataset = None
    assert px_m == pytest.approx(5.0 * 0.3048, rel=1e-3)
    assert py_m == pytest.approx(5.0 * 0.3048, rel=1e-3)


def test_compute_slope_deg_flat_plane_is_zero():
    heights = np.full((10, 10), 42.0, dtype=np.float32)
    slope = lidar_masks.compute_slope_deg(heights, 1.5, 1.5)
    np.testing.assert_allclose(slope, 0.0, atol=1e-4)


def test_compute_slope_deg_matches_known_ramp():
    # height increases 1m per 1m of x -> 45 deg slope everywhere except the
    # array edges (one-sided differences there are still 1:1, so it holds
    # across the whole array for a perfectly linear ramp).
    x = np.arange(20, dtype=np.float32)
    heights = np.tile(x, (20, 1))
    slope = lidar_masks.compute_slope_deg(heights, 1.0, 1.0)
    np.testing.assert_allclose(slope, 45.0, atol=1e-3)


def test_compute_horizontality_excluded_flags_steep_not_flat():
    heights = np.zeros((10, 10), dtype=np.float32)
    heights[:, 5:] = 100.0  # a sharp step -> near-vertical between columns 4 and 5
    valid = np.ones((10, 10), dtype=bool)
    excluded = lidar_masks.compute_horizontality_excluded(heights, valid, 1.5, 1.5, max_slope_deg=5.0)
    assert excluded[:, 4:6].all()  # the step itself is excluded
    assert not excluded[:, 0].any()  # flat interior far from the step is not
    assert not excluded[:, -1].any()


def test_compute_horizontality_excluded_never_excludes_invalid_pixels():
    heights = np.zeros((6, 6), dtype=np.float32)
    heights[:, 3:] = 999.0
    valid = np.zeros((6, 6), dtype=bool)  # nothing has real DSM coverage
    excluded = lidar_masks.compute_horizontality_excluded(heights, valid, 1.5, 1.5, max_slope_deg=5.0)
    assert not excluded.any()


def test_roughness_distinguishes_smooth_slope_from_rough_surface():
    # A perfectly smooth (but steep) ramp: every normal vector points the
    # same direction -> roughness ~ 0 despite a large slope. VRM should NOT
    # flag this as rough even though horizontality would flag it as not flat.
    x = np.arange(30, dtype=np.float32)
    smooth_ramp = np.tile(x * 2.0, (30, 1))
    valid = np.ones((30, 30), dtype=bool)
    roughness_smooth = lidar_masks.compute_roughness_deg(smooth_ramp, 1.0, 1.0, window_radius_px=2)
    # arccos of a resultant length extremely close to 1.0 is numerically
    # sensitive (its derivative blows up there), so a perfectly uniform
    # slope still comes back as a small nonzero angle rather than exact 0 --
    # still two orders of magnitude below the noisy-surface comparison below.
    np.testing.assert_allclose(roughness_smooth[5:-5, 5:-5], 0.0, atol=0.05)

    # A checkerboard of alternating heights: normals point in wildly
    # different directions within any window -> high roughness.
    rng = np.random.default_rng(0)
    noisy = rng.uniform(0.0, 5.0, size=(30, 30)).astype(np.float32)
    roughness_noisy = lidar_masks.compute_roughness_deg(noisy, 1.0, 1.0, window_radius_px=2)
    assert roughness_noisy[5:-5, 5:-5].mean() > roughness_smooth[5:-5, 5:-5].mean() + 10.0


def test_compute_roughness_excluded_respects_threshold_and_erodes_invalid_border():
    rng = np.random.default_rng(1)
    noisy = rng.uniform(0.0, 5.0, size=(20, 20)).astype(np.float32)
    valid = np.ones((20, 20), dtype=bool)
    valid[0, :] = False  # a strip of missing DSM coverage along one edge
    excluded = lidar_masks.compute_roughness_excluded(
        noisy, valid, 1.0, 1.0, window_radius_px=2, max_roughness_deg=1.0,
    )
    # A single square (2*2+1)-wide erosion pass removes validity exactly 2
    # pixels deep from row 0's invalidity -- rows 0-2 must never be
    # excluded regardless of their independently-computed roughness value.
    assert not excluded[:3, :].any()
    # ...but the strict 1 deg threshold on uniform random noise does flag
    # genuine roughness once far enough from the border to be unaffected by
    # the erosion -- otherwise this test would pass vacuously.
    assert excluded[10:, :].any()


def test_compute_shadow_excluded_tall_obstruction_casts_shadow():
    # A 3m obstruction at 45 deg sun elevation casts a shadow exactly 3m
    # (== 3 pixels at 1m spacing) long: shadow_length = height / tan(elevation).
    heights = np.zeros((40, 40), dtype=np.float32)
    heights[20, 20] = 3.0
    valid = np.ones((40, 40), dtype=bool)
    # Sun due east (azimuth 90) -> shadow cast westward from the obstruction.
    excluded = lidar_masks.compute_shadow_excluded(
        heights, valid, 1.0, 1.0, azimuth_deg=90.0, elevation_deg=45.0,
        max_building_height_m=10.0, local_avg_radius_px=0,
    )
    assert excluded[20, 18]  # 2m upwind: within the 3m shadow reach
    assert not excluded[20, 5]  # 15m upwind: far outside the 3m shadow reach
    assert not excluded[0, 5]  # unrelated row, nowhere near the obstruction: unaffected


def test_compute_shadow_excluded_sun_below_horizon_shadows_everything():
    heights = np.zeros((5, 5), dtype=np.float32)
    valid = np.ones((5, 5), dtype=bool)
    excluded = lidar_masks.compute_shadow_excluded(heights, valid, 1.0, 1.0, azimuth_deg=180.0, elevation_deg=0.0)
    assert excluded.all()


def test_downsample_height_window_block_averages():
    heights = np.array([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]] * 2, dtype=np.float32)
    valid = np.ones_like(heights, dtype=bool)
    info = io.RasterInfo(path="x", width=4, height=4, band_count=1, crs="", geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), nodata=None)
    window = lidar_masks.DsmWindow(heights_m=heights, valid=valid, info=info)
    down = lidar_masks.downsample_height_window(window, 2)
    assert down.heights_m.shape == (2, 2)
    assert down.info.geotransform[1] == 2.0
    assert down.info.geotransform[5] == -2.0
    np.testing.assert_allclose(down.heights_m[0, 0], np.mean([1.0, 3.0, 2.0, 4.0]))


def test_downsample_height_window_factor_one_is_noop():
    heights = np.arange(9, dtype=np.float32).reshape(3, 3)
    info = io.RasterInfo(path="x", width=3, height=3, band_count=1, crs="", geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), nodata=None)
    window = lidar_masks.DsmWindow(heights_m=heights, valid=np.ones((3, 3), dtype=bool), info=info)
    assert lidar_masks.downsample_height_window(window, 1) is window


def test_read_dsm_window_for_scene_crops_and_converts_units(tmp_path):
    heights_ft = np.full((20, 20), 10.0, dtype=np.float32)  # 10 ft everywhere
    dsm_path = _write_dsm(str(tmp_path / "dsm.tif"), heights_ft, origin=(0.0, 100.0), pixel_size=1.0)
    dataset = lidar_masks.open_dsm_mosaic(dsm_path)

    scene_path = write_geotiff(
        str(tmp_path / "scene.tif"), [np.zeros((5, 5), dtype=np.uint16)],
        origin=(5.0, 95.0), pixel_size=1.0,
    )
    scene_info = io.get_raster_info(scene_path)

    window = lidar_masks.read_dsm_window_for_scene(dataset, scene_info, height_units="ft", buffer_m=2.0)
    assert window is not None
    np.testing.assert_allclose(window.heights_m, 10.0 * 0.3048, atol=1e-3)
    assert window.valid.all()


def test_read_dsm_window_for_scene_returns_none_when_disjoint(tmp_path):
    heights = np.zeros((5, 5), dtype=np.float32)
    dsm_path = _write_dsm(str(tmp_path / "dsm.tif"), heights, origin=(0.0, 100.0), pixel_size=1.0)
    dataset = lidar_masks.open_dsm_mosaic(dsm_path)

    scene_path = write_geotiff(
        str(tmp_path / "scene.tif"), [np.zeros((5, 5), dtype=np.uint16)],
        origin=(10000.0, 10000.0), pixel_size=1.0,
    )
    scene_info = io.get_raster_info(scene_path)
    window = lidar_masks.read_dsm_window_for_scene(dataset, scene_info, buffer_m=0.0)
    assert window is None


def test_read_dsm_window_for_scenes_covers_union_not_just_first_scene(tmp_path):
    # A DSM wide enough to cover two non-overlapping scene footprints side
    # by side. read_dsm_window_for_scenes must read enough of the DSM to
    # cover BOTH -- read_dsm_window_for_scene (singular), given only the
    # first scene, would miss the second's area entirely, which is exactly
    # the bug this covers: a target scene extending beyond the reference's
    # own footprint getting zero LiDAR coverage there.
    heights = np.arange(400, dtype=np.float32).reshape(20, 20)
    dsm_path = _write_dsm(str(tmp_path / "dsm.tif"), heights, origin=(0.0, 100.0), pixel_size=1.0)
    dataset = lidar_masks.open_dsm_mosaic(dsm_path)

    scene_a_path = write_geotiff(
        str(tmp_path / "scene_a.tif"), [np.zeros((4, 4), dtype=np.uint16)], origin=(1.0, 99.0), pixel_size=1.0,
    )
    scene_b_path = write_geotiff(
        str(tmp_path / "scene_b.tif"), [np.zeros((4, 4), dtype=np.uint16)], origin=(15.0, 85.0), pixel_size=1.0,
    )
    info_a = io.get_raster_info(scene_a_path)
    info_b = io.get_raster_info(scene_b_path)

    single = lidar_masks.read_dsm_window_for_scene(dataset, info_a, buffer_m=0.0)
    union = lidar_masks.read_dsm_window_for_scenes(dataset, [info_a, info_b], buffer_m=0.0)

    assert single is not None and union is not None
    assert single.heights_m.shape == (4, 4)
    # the union window must be large enough to span from scene_a's origin
    # out to scene_b's far corner -- strictly bigger than either scene alone.
    assert union.heights_m.shape[0] > single.heights_m.shape[0]
    assert union.heights_m.shape[1] > single.heights_m.shape[1]


def test_compute_and_cache_slope_roughness_matches_whole_array_across_block_boundary(tmp_path):
    # block_rows smaller than the DSM height forces multiple blocks with a
    # halo-stitched boundary between them -- results must be pixel-identical
    # to computing slope/roughness on the whole array at once (the tiny
    # unblocked helpers exercised by the other tests in this file), which is
    # only true if the halo padding at internal block boundaries is real
    # neighboring DSM data, not a synthetic edge repeat.
    rng = np.random.default_rng(3)
    heights = rng.uniform(0.0, 30.0, size=(50, 12)).astype(np.float32)
    dsm_path = _write_dsm(str(tmp_path / "dsm.tif"), heights, origin=(0.0, 100.0), pixel_size=1.0)
    dataset = lidar_masks.open_dsm_mosaic(dsm_path)

    scene_path = write_geotiff(
        str(tmp_path / "scene.tif"), [np.zeros((50, 12), dtype=np.uint16)], origin=(0.0, 100.0), pixel_size=1.0,
    )
    scene_info = io.get_raster_info(scene_path)

    slope_path = str(tmp_path / "slope.tif")
    rough_path = str(tmp_path / "rough.tif")
    out_info = lidar_masks.compute_and_cache_slope_roughness(
        dataset, [scene_info], buffer_m=0.0,
        max_slope_deg=5.0, roughness_window_radius_px=2, roughness_max_deg=10.0,
        slope_path=slope_path, roughness_path=rough_path, block_rows=7,  # forces ~7 blocks over 50 rows
    )
    assert out_info is not None and out_info.height == 50 and out_info.width == 12

    def read(p):
        ds = gdal.Open(p)
        a = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        return a

    blocked_slope = read(slope_path) != 0
    blocked_rough = read(rough_path) != 0

    valid = np.ones((50, 12), dtype=bool)
    expected_slope = lidar_masks.compute_horizontality_excluded(heights, valid, 1.0, 1.0, max_slope_deg=5.0)
    expected_rough = lidar_masks.compute_roughness_excluded(heights, valid, 1.0, 1.0, window_radius_px=2, max_roughness_deg=10.0)

    np.testing.assert_array_equal(blocked_slope, expected_slope)
    np.testing.assert_array_equal(blocked_rough, expected_rough)


def test_compute_and_cache_shadow_matches_whole_array_across_block_boundary(tmp_path):
    # block_rows smaller than the DSM height forces multiple blocks with a
    # halo-stitched boundary -- results must be pixel-identical to computing
    # shadow on the whole array at once, which is only true if the halo
    # padding (sized to the full shadow search radius, not a small window)
    # at internal block boundaries is real neighboring DSM data.
    heights = np.zeros((60, 15), dtype=np.float32)
    heights[30, 10] = 3.0  # a small obstruction partway down/across
    dsm_path = _write_dsm(str(tmp_path / "dsm.tif"), heights, origin=(0.0, 100.0), pixel_size=1.0)
    dataset = lidar_masks.open_dsm_mosaic(dsm_path)

    scene_path = write_geotiff(
        str(tmp_path / "scene.tif"), [np.zeros((60, 15), dtype=np.uint16)], origin=(0.0, 100.0), pixel_size=1.0,
    )
    scene_info = io.get_raster_info(scene_path)

    shadow_path = str(tmp_path / "shadow.tif")
    azimuth_deg, elevation_deg = 90.0, 45.0  # 3m obstruction -> 3m/3px shadow reach
    out_info = lidar_masks.compute_and_cache_shadow(
        dataset, [scene_info], azimuth_deg=azimuth_deg, elevation_deg=elevation_deg,
        max_building_height_m=10.0, shadow_path=shadow_path, block_rows=7,  # forces ~9 blocks over 60 rows
    )
    assert out_info is not None and out_info.height == 60 and out_info.width == 15

    ds = gdal.Open(shadow_path)
    blocked = ds.GetRasterBand(1).ReadAsArray() != 0
    ds = None

    valid = np.ones((60, 15), dtype=bool)
    expected = lidar_masks.compute_shadow_excluded(
        heights, valid, 1.0, 1.0, azimuth_deg, elevation_deg, max_building_height_m=10.0,
    )
    np.testing.assert_array_equal(blocked, expected)
    assert expected.any()  # sanity: the obstruction actually casts a shadow somewhere


def test_compute_and_cache_shadow_leaves_no_file_at_final_path_if_interrupted(tmp_path, monkeypatch):
    # A crash partway through a many-hour run must not leave a partially-
    # written file at shadow_path -- a resumed run's os.path.exists check
    # would otherwise wrongly treat an incomplete file as a finished cache.
    heights = np.zeros((30, 10), dtype=np.float32)
    dsm_path = _write_dsm(str(tmp_path / "dsm.tif"), heights, origin=(0.0, 100.0), pixel_size=1.0)
    dataset = lidar_masks.open_dsm_mosaic(dsm_path)
    scene_path = write_geotiff(
        str(tmp_path / "scene.tif"), [np.zeros((30, 10), dtype=np.uint16)], origin=(0.0, 100.0), pixel_size=1.0,
    )
    scene_info = io.get_raster_info(scene_path)
    shadow_path = str(tmp_path / "shadow.tif")

    call_count = {"n": 0}
    real_compute = lidar_masks.compute_shadow_excluded

    def _flaky(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated crash mid-run")
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(lidar_masks, "compute_shadow_excluded", _flaky)

    with pytest.raises(RuntimeError, match="simulated crash"):
        lidar_masks.compute_and_cache_shadow(
            dataset, [scene_info], azimuth_deg=90.0, elevation_deg=45.0,
            max_building_height_m=10.0, shadow_path=shadow_path, block_rows=5,  # forces multiple blocks
        )
    assert not os.path.exists(shadow_path)


def test_compute_and_cache_slope_roughness_returns_none_when_disjoint(tmp_path):
    heights = np.zeros((5, 5), dtype=np.float32)
    dsm_path = _write_dsm(str(tmp_path / "dsm.tif"), heights, origin=(0.0, 100.0), pixel_size=1.0)
    dataset = lidar_masks.open_dsm_mosaic(dsm_path)

    scene_path = write_geotiff(
        str(tmp_path / "scene.tif"), [np.zeros((5, 5), dtype=np.uint16)], origin=(10000.0, 10000.0), pixel_size=1.0,
    )
    scene_info = io.get_raster_info(scene_path)
    out_info = lidar_masks.compute_and_cache_slope_roughness(
        dataset, [scene_info], buffer_m=0.0,
        max_slope_deg=5.0, roughness_window_radius_px=2, roughness_max_deg=10.0,
        slope_path=str(tmp_path / "slope.tif"), roughness_path=str(tmp_path / "rough.tif"),
    )
    assert out_info is None


def test_open_dsm_mosaic_builds_vrt_from_tile_directory(tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    _write_dsm(str(tiles_dir / "a.tif"), np.zeros((5, 5), dtype=np.float32), origin=(0.0, 100.0), pixel_size=1.0)
    _write_dsm(str(tiles_dir / "b.tif"), np.ones((5, 5), dtype=np.float32), origin=(5.0, 100.0), pixel_size=1.0)
    dataset = lidar_masks.open_dsm_mosaic(str(tiles_dir))
    assert dataset.RasterXSize == 10
    assert dataset.RasterYSize == 5


def test_warp_mask_to_scene_conservatively_includes_flagged_subpixels(tmp_path):
    # DSM-grid mask at 1m, one flagged pixel among 4x4; scene grid is 2x
    # coarser (2m) covering the same extent -- the coarse output pixel that
    # contains the flagged fine pixel should come back True (max-resample).
    #
    # Uses a full ImportFromEPSG-derived WKT rather than this file's usual
    # hand-typed PROJCS string (see write_geotiff in _raster_utils.py):
    # that string omits the actual Transverse Mercator parameters (central
    # meridian, scale factor, false easting/northing), which every other
    # test tolerates since nothing else here invokes real PROJ-based
    # reprojection -- but gdal.Warp does, and an underspecified projection
    # makes it reproject incorrectly even between "the same" CRS.
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32618)
    crs = srs.ExportToWkt()
    dsm_info = io.RasterInfo(
        path="x", width=4, height=4, band_count=1, crs=crs,
        geotransform=(0.0, 1.0, 0.0, 100.0, 0.0, -1.0), nodata=None,
    )
    scene_info = io.RasterInfo(
        path="y", width=2, height=2, band_count=1, crs=crs,
        geotransform=(0.0, 2.0, 0.0, 100.0, 0.0, -2.0), nodata=None,
    )
    mask_path = str(tmp_path / "mask.tif")
    masking.save_flags_raster(mask, dsm_info, mask_path)

    out = lidar_masks.warp_mask_to_scene(mask_path, scene_info)
    assert out.shape == (2, 2)
    assert out[0, 0]  # top-left coarse pixel covers DSM rows/cols 0-1, where (1,1) is flagged
    assert not out[1, 1]
