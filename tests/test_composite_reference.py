"""Tests for composite_reference.py against a small synthetic composite
(built in-memory via GDAL, mirroring compositing.py's band-naming
convention and NODATA), so these don't depend on any real multi-GB data
being present on disk.
"""

import os

import numpy as np
import pytest
from osgeo import gdal

from psnorm import composite_reference as cref
from psnorm import io

gdal.UseExceptions()

NODATA = -9999.0
CANONICAL_BANDS = ["blue", "green", "red", "nir"]
STATS = ["min", "median", "max"]


def _make_composite(path: str, width: int = 200, height: int = 200, origin=(500000.0, 4400000.0), pixel_size: float = 3.0):
    band_names = [f"{b}_{s}" for b in CANONICAL_BANDS for s in STATS]
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, width, height, len(band_names), gdal.GDT_Float32)
    ds.SetGeoTransform((origin[0], pixel_size, 0.0, origin[1], 0.0, -pixel_size))
    srs_wkt = (
        'PROJCS["WGS 84 / UTM zone 18N",GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
        'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-75],'
        'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
        'PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","32618"]]'
    )
    ds.SetProjection(srs_wkt)
    rng = np.random.default_rng(0)
    for i, name in enumerate(band_names, start=1):
        band = ds.GetRasterBand(i)
        band.SetDescription(name)
        band.SetNoDataValue(NODATA)
        arr = rng.uniform(100, 2000, size=(height, width)).astype(np.float32)
        arr[0, 0] = NODATA  # a token nodata pixel
        band.WriteArray(arr)
    ds = None
    return band_names


@pytest.fixture
def composite_path(tmp_path):
    path = str(tmp_path / "solstice_test.tif")
    _make_composite(path)
    return path


def test_is_composite_true_for_multi_stat_raster(composite_path):
    assert cref.is_composite(composite_path)


def test_is_composite_false_for_plain_scene(tmp_path):
    path = str(tmp_path / "scene.tif")
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, 10, 10, 4, gdal.GDT_Float32)
    for i, name in enumerate(["blue", "green", "red", "nir"], start=1):
        ds.GetRasterBand(i).SetDescription(name)
    ds = None
    assert not cref.is_composite(path)


def test_ensure_single_stat_cache_has_expected_bands(composite_path):
    cache_path = cref.ensure_single_stat_cache(composite_path, stat="median")
    assert os.path.exists(cache_path)
    names = cref.composite_band_names(cache_path)
    assert names == [f"{b}_median" for b in CANONICAL_BANDS]


def test_ensure_single_stat_cache_is_reused(composite_path):
    cache_path_1 = cref.ensure_single_stat_cache(composite_path, stat="median")
    mtime_1 = os.path.getmtime(cache_path_1)
    cache_path_2 = cref.ensure_single_stat_cache(composite_path, stat="median")
    assert cache_path_1 == cache_path_2
    assert os.path.getmtime(cache_path_2) == mtime_1  # not rebuilt


def test_extract_reference_window_shape_and_nodata(composite_path):
    # A small target window fully inside the composite's extent.
    target_info = io.RasterInfo(
        path="fake_target.tif", width=10, height=10, band_count=4,
        crs="", geotransform=(500030.0, 3.0, 0.0, 4399970.0, 0.0, -3.0), nodata=None,
    )
    # Target is the 10x10 px block starting at composite px (10, 10); a 15 m
    # buffer at 3 m pixels is 5 px each side -> a 20x20 window at (5, 5).
    result = cref.extract_reference_window(composite_path, target_info, CANONICAL_BANDS, stat="median", buffer_m=15.0)
    assert result.shape == (len(CANONICAL_BANDS), 20, 20)
    assert result.rio.crs is not None
    assert result.rio.transform().c == 500000.0 + 5 * 3.0
    assert result.rio.transform().f == 4400000.0 - 5 * 3.0
    assert np.isnan(result.values).sum() == 0  # the token nodata pixel at (0, 0) is outside this window


def test_extract_reference_window_buffer_clamped_to_composite_bounds(composite_path):
    target_info = io.RasterInfo(
        path="fake_target.tif", width=10, height=10, band_count=4,
        crs="", geotransform=(500030.0, 3.0, 0.0, 4399970.0, 0.0, -3.0), nodata=None,
    )
    # 500 m = 167 px, far more than the 10 px available above/left of the
    # target; the window should clamp to the composite edge, not error.
    result = cref.extract_reference_window(composite_path, target_info, CANONICAL_BANDS, stat="median", buffer_m=500.0)
    assert result.shape == (len(CANONICAL_BANDS), 187, 187)
    assert result.rio.transform().c == 500000.0
    assert np.isnan(result.values[:, 0, 0]).all()  # composite nodata converted to NaN


def test_extract_reference_window_missing_stat_raises(composite_path):
    target_info = io.RasterInfo(
        path="fake_target.tif", width=10, height=10, band_count=4,
        crs="", geotransform=(500030.0, 3.0, 0.0, 4399970.0, 0.0, -3.0), nodata=None,
    )
    with pytest.raises(ValueError):
        cref.extract_reference_window(composite_path, target_info, CANONICAL_BANDS, stat="p90")


def test_extract_reference_window_no_overlap_raises(composite_path):
    # Far away from the composite's extent.
    target_info = io.RasterInfo(
        path="fake_target.tif", width=10, height=10, band_count=4,
        crs="", geotransform=(900000.0, 3.0, 0.0, 4000000.0, 0.0, -3.0), nodata=None,
    )
    with pytest.raises(ValueError):
        cref.extract_reference_window(composite_path, target_info, CANONICAL_BANDS, stat="median")


# --------------------------------------------------------------------------
# Composites as normalization references
# --------------------------------------------------------------------------

EIGHT_BANDS = ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"]


def _make_eight_band_composite(path, *, eight_band_data: bool, size=40):
    """A median-only composite with all 8 canonical bands; without
    `eight_band_data`, the 8-band-only bands are empty (all nodata), as in
    a composite built before 8-band scenes existed."""
    ds = gdal.GetDriverByName("GTiff").Create(path, size, size, len(EIGHT_BANDS), gdal.GDT_Float32)
    ds.SetGeoTransform((500000.0, 3.0, 0.0, 4400000.0, 0.0, -3.0))
    rng = np.random.default_rng(1)
    for i, name in enumerate(EIGHT_BANDS, start=1):
        band = ds.GetRasterBand(i)
        band.SetDescription(f"{name}_median")
        band.SetNoDataValue(NODATA)
        arr = rng.uniform(100, 2000, (size, size)).astype(np.float32)
        arr[:, :4] = NODATA  # an edge outside the AOI, in every band
        if not eight_band_data and name in cref.EIGHT_BAND_ONLY:
            arr[:] = NODATA
        band.WriteArray(arr)
    ds = None
    return path


def test_band_coverage_and_present_bands_drop_empty_eight_band_layers(tmp_path):
    early = _make_eight_band_composite(str(tmp_path / "solstice_2019.tif"), eight_band_data=False)
    late = _make_eight_band_composite(str(tmp_path / "solstice_2023.tif"), eight_band_data=True)
    early_coverage = cref.band_coverage(early)
    assert early_coverage["blue"] == pytest.approx(0.9, abs=0.01)
    assert early_coverage["coastal_blue"] == 0.0
    assert cref.present_bands(early_coverage) == ["blue", "green", "red", "nir"]
    assert cref.present_bands(cref.band_coverage(late)) == EIGHT_BANDS


def test_ensure_reference_raster_names_bands_plainly(tmp_path):
    from psnorm import sensors
    path = _make_eight_band_composite(str(tmp_path / "solstice_2023.tif"), eight_band_data=True)
    vrt = cref.ensure_reference_raster(path, ["blue", "red", "nir"], str(tmp_path / "ref"))
    assert sensors.detect_band_names(vrt) == ["blue", "red", "nir"]
    cache = cref.ensure_single_stat_cache(path)
    np.testing.assert_array_equal(io.read_window_bands(vrt, (0, 0, 40, 40))[1],
                                  io.read_window_bands(cache, (0, 0, 40, 40))[EIGHT_BANDS.index("red")])
    assert io.get_raster_info(vrt).nodata == NODATA


def test_find_solstice_composites(tmp_path):
    for name in ("solstice_2021.tif", "solstice_2019.tif", "solstice_2019.tif.aux.xml", "other_2020.tif"):
        (tmp_path / name).write_bytes(b"")
    composites = cref.find_solstice_composites(str(tmp_path), window_days=14)
    assert [c.year for c in composites] == [2019, 2021]
    assert composites[0].name == "solstice_2019"
    assert composites[0].solstice.month == 6 and composites[0].solstice.day in (20, 21, 22)
    assert composites[0].window_start == composites[0].solstice - __import__("datetime").timedelta(days=14)


def _scene(scene_id):
    from psnorm.io import Scene, parse_acquisition_time
    return Scene(scene_id, f"/x/{scene_id}.tif", None, None, parse_acquisition_time(scene_id))


def test_assign_scenes_to_composites_keeps_eight_band_scenes_after_the_onset():
    from datetime import datetime
    composites = [cref.ReferenceComposite(f"/c/solstice_{y}.tif", y, datetime(y, 6, 21), 14) for y in (2020, 2021, 2022, 2023)]
    four, eight = ["blue", "green", "red", "nir"], EIGHT_BANDS
    # As in the Philadelphia composites: only 2022 on carry the 8-band-only bands.
    composite_bands = {c.path: (eight if c.year >= 2022 else four) for c in composites}
    scenes = {
        "20210601_150000_a": four,    # 4-band: nearest composite, 2021
        "20210905_150000_b": eight,   # 8-band, right after the onset: 2021 lacks its bands, so 2022
        "20230701_150000_c": eight,   # nearest, 2023
        "20200101_150000_d": four,    # nearest, 2020
    }
    objs = [_scene(s) for s in scenes]
    onset = cref.eight_band_onset(scenes, objs)
    assert onset == datetime(2021, 9, 5, 15)
    groups = cref.assign_scenes_to_composites(objs, scenes, composites, composite_bands, onset=onset)
    chosen = {s.scene_id: path for path, group in groups.items() for s in group}
    assert chosen == {"20210601_150000_a": "/c/solstice_2021.tif", "20210905_150000_b": "/c/solstice_2022.tif",
                      "20230701_150000_c": "/c/solstice_2023.tif", "20200101_150000_d": "/c/solstice_2020.tif"}

    # The onset rule holds even when a composite has every band: with all
    # composites 8-band, a 2021 composite (window from June 7) still predates
    # an 8-band onset of September 2021.
    all_eight = {c.path: eight for c in composites}
    groups = cref.assign_scenes_to_composites(objs, scenes, composites, all_eight, onset=onset)
    chosen = {s.scene_id: path for path, group in groups.items() for s in group}
    assert chosen["20210905_150000_b"] == "/c/solstice_2022.tif"
    assert chosen["20210601_150000_a"] == "/c/solstice_2021.tif"

    # A fixed year, used wherever it is eligible.
    groups = cref.assign_scenes_to_composites(objs, scenes, composites, composite_bands, onset=onset, fixed_year=2020)
    chosen = {s.scene_id: path for path, group in groups.items() for s in group}
    assert chosen["20210601_150000_a"] == "/c/solstice_2020.tif"
    assert chosen["20230701_150000_c"] == "/c/solstice_2023.tif"


def test_scene_band_names_from_file_names():
    from psnorm.io import Scene
    scenes = [Scene("a", "/r/a_3B_AnalyticMS_SR_harmonized_clip.tif", None, None, None),
              Scene("b", "/r/b_3B_AnalyticMS_SR_8b_harmonized_clip.tif", None, None, None)]
    assert cref.scene_band_names(scenes) == {"a": ["blue", "green", "red", "nir"], "b": EIGHT_BANDS}
