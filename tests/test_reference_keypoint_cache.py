"""composite_reference's stored composite keypoints: built once, and a
window load returns exactly the stored keypoints that fall in the window."""

import json
import os

import cv2
import numpy as np
import pytest
from osgeo import gdal

from psnorm import composite_reference

gdal.UseExceptions()

GEOTRANSFORM = (500000.0, 3.0, 0.0, 4400000.0, 0.0, -3.0)
WIDTH, HEIGHT = 1400, 1100
BANDS = ["blue", "green", "red", "nir"]


@pytest.fixture(scope="module")
def composite(tmp_path_factory):
    """A small composite with textured `_median` bands (and one other
    statistic, which must be ignored)."""
    path = str(tmp_path_factory.mktemp("composite") / "solstice_2024.tif")
    rng = np.random.default_rng(0)
    names = [f"{b}_median" for b in BANDS] + ["blue_p75"]
    ds = gdal.GetDriverByName("GTiff").Create(path, WIDTH, HEIGHT, len(names), gdal.GDT_Float32)
    ds.SetGeoTransform(GEOTRANSFORM)
    ds.SetProjection('PROJCS["WGS 84 / UTM zone 18N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
                     '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION['
                     '"Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-75],'
                     'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],'
                     'UNIT["metre",1]]')
    for i, name in enumerate(names, start=1):
        # Independent, full-range texture per band, so that the normalized
        # differences between bands span most of [-1, 1].
        texture = cv2.GaussianBlur(rng.random((HEIGHT, WIDTH)).astype(np.float32), (0, 0), 3)
        texture = 100 + 4000 * (texture - texture.min()) / (texture.max() - texture.min())
        band = ds.GetRasterBand(i)
        band.WriteArray(texture)
        band.SetDescription(name)
    ds = None
    return path


def test_keypoints_are_built_once_and_stored_per_setting(composite):
    first = composite_reference.ensure_reference_keypoints(composite, BANDS, detection_scale=1.0)
    with open(os.path.join(first, "meta.json")) as f:
        meta = json.load(f)
    assert meta["bands"] == BANDS and meta["detection_scale"] == 1.0
    assert sorted(meta["keypoints"]) == sorted(
        ["blue-green", "blue-red", "blue-nir", "green-red", "green-nir", "red-nir"])
    assert all(count > 50 for count in meta["keypoints"].values())

    stamp = os.path.getmtime(os.path.join(first, "meta.json"))
    assert composite_reference.ensure_reference_keypoints(composite, BANDS, detection_scale=1.0) == first
    assert os.path.getmtime(os.path.join(first, "meta.json")) == stamp  # not rebuilt

    # A different scale, band order or blur setting is its own store.
    others = {
        composite_reference.ensure_reference_keypoints(composite, BANDS, detection_scale=0.5),
        composite_reference.ensure_reference_keypoints(composite, BANDS[::-1], detection_scale=1.0),
        composite_reference.ensure_reference_keypoints(composite, BANDS, detection_scale=1.0, blur_kernel_base_size=5),
    }
    assert len(others | {first}) == 4


def test_window_load_returns_the_stored_keypoints_inside_the_window(composite):
    keypoint_dir = composite_reference.ensure_reference_keypoints(composite, BANDS, detection_scale=1.0)
    left, top = GEOTRANSFORM[0], GEOTRANSFORM[3]
    everything = composite_reference.load_reference_keypoints(
        keypoint_dir, (left, top - HEIGHT * 3.0, left + WIDTH * 3.0, top))
    # Columns 300-900, rows 200-700 (not aligned to the storage cells).
    window = (left + 300 * 3.0, top - 700 * 3.0, left + 900 * 3.0, top - 200 * 3.0)
    part = composite_reference.load_reference_keypoints(keypoint_dir, window)

    assert sorted(everything.channels) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert tuple(everything.transform.to_gdal()) == GEOTRANSFORM
    for key, full in everything.channels.items():
        def inside(points):
            return (points[:, 0] >= 300) & (points[:, 0] < 900) & (points[:, 1] >= 200) & (points[:, 1] < 700)

        expected = inside(full.points)
        got = inside(part.channels[key].points)
        assert expected.sum() > 10
        # Same keypoints, with their own descriptors, whatever the order.
        def keyed(features, keep):
            rows = np.column_stack([features.points[keep], features.descriptors[keep].astype(np.float32)])
            return rows[np.lexsort(rows.T[::-1])]

        np.testing.assert_array_equal(keyed(part.channels[key], got), keyed(full, expected))
        # ...and far fewer were read than the whole composite holds.
        assert len(part.channels[key].points) < len(full.points)


def test_cache_dir_and_compression_change_where_and_how_but_not_what(composite, tmp_path):
    """The single-statistic copy in another directory, uncompressed, holds
    the same pixels, and windows read through it are identical."""
    from psnorm import io

    default_path = composite_reference.ensure_single_stat_cache(composite)
    other_path = composite_reference.ensure_single_stat_cache(
        composite, cache_dir=str(tmp_path / "fast"), compression="NONE")
    assert os.path.dirname(other_path) == str(tmp_path / "fast")
    assert os.path.basename(other_path) == os.path.basename(default_path)
    assert gdal.Open(default_path).GetMetadata("IMAGE_STRUCTURE").get("COMPRESSION") == "LZW"
    assert "COMPRESSION" not in gdal.Open(other_path).GetMetadata("IMAGE_STRUCTURE")
    np.testing.assert_array_equal(gdal.Open(default_path).ReadAsArray(), gdal.Open(other_path).ReadAsArray())

    target = io.RasterInfo(path="t.tif", width=300, height=200, band_count=4, crs=io.get_raster_info(composite).crs,
                           geotransform=(GEOTRANSFORM[0] + 600.0, 3.0, 0.0, GEOTRANSFORM[3] - 900.0, 0.0, -3.0),
                           nodata=None)
    a = composite_reference.extract_reference_window(composite, target, BANDS, buffer_m=30.0)
    b = composite_reference.extract_reference_window(composite, target, BANDS, buffer_m=30.0,
                                                     cache_dir=str(tmp_path / "fast"), cache_compression="NONE")
    np.testing.assert_array_equal(a.values, b.values)
    assert a.rio.transform() == b.rio.transform()

    # Stored keypoints go with the copy they were detected on.
    keypoint_dir = composite_reference.ensure_reference_keypoints(
        composite, BANDS, detection_scale=0.5, cache_dir=str(tmp_path / "fast"), cache_compression="NONE")
    assert os.path.dirname(keypoint_dir) == str(tmp_path / "fast")
