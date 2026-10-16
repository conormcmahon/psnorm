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
