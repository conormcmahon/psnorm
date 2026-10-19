import json

import numpy as np
from osgeo import gdal

from psnorm import io, masking

from _raster_utils import write_geotiff


def test_compute_exclusion_flags_combines_nodata_water_and_udm2(tmp_path):
    size = (4, 4)
    blue = np.full(size, 150, dtype=np.uint16)
    green = np.full(size, 120, dtype=np.uint16)
    red = np.full(size, 130, dtype=np.uint16)
    nir = np.full(size, 400, dtype=np.uint16)  # land: green << nir -> NDWI well below 0

    blue[0, 0] = green[0, 0] = red[0, 0] = nir[0, 0] = 0  # nodata pixel

    green[2, 2] = 300  # water pixel: green >> nir -> NDWI > 0
    nir[2, 2] = 50

    analytic_path = write_geotiff(
        str(tmp_path / "a.tif"), [blue, green, red, nir],
        band_descriptions=["blue", "green", "red", "nir"], nodata=0,
    )

    udm2_clear = np.ones(size, dtype=np.uint8)
    udm2_clear[1, 1] = 0  # not-clear per UDM2
    udm2_path = write_geotiff(str(tmp_path / "a_udm2.tif"), [udm2_clear] + [np.ones(size, dtype=np.uint8)] * 7,
                               nodata=None)

    flags = masking.compute_exclusion_flags(
        analytic_path, udm2_path, ["blue", "green", "red", "nir"],
        use_water_mask=True, ndwi_threshold=0.0, use_vegetation_mask=False, use_omnicloudmask=False,
    )
    assert flags.shape == size
    assert flags[0, 0] & masking.EXCLUDE_NODATA
    assert flags[1, 1] & masking.EXCLUDE_UDM2
    assert flags[2, 2] & masking.EXCLUDE_WATER
    assert flags[3, 3] == 0  # otherwise-clear land pixel: no exclusion bits set


def test_compute_ndwi_water_mask(tmp_path):
    green = np.array([[300, 100]], dtype=np.uint16)  # water, land
    nir = np.array([[50, 300]], dtype=np.uint16)
    blue = np.full((1, 2), 100, dtype=np.uint16)
    red = np.full((1, 2), 100, dtype=np.uint16)
    path = write_geotiff(str(tmp_path / "a.tif"), [blue, green, red, nir],
                          band_descriptions=["blue", "green", "red", "nir"], nodata=0)

    water = masking.compute_ndwi_water_mask(path, (0, 0, 2, 1), ["blue", "green", "red", "nir"], ndwi_threshold=0.0)
    assert list(water[0]) == [True, False]


def test_compute_ndvi_vegetation_mask(tmp_path):
    red = np.array([[100, 100]], dtype=np.uint16)  # vegetation, bare/land
    nir = np.array([[400, 110]], dtype=np.uint16)  # NDVI = 0.6, NDVI ~ 0.048
    blue = np.full((1, 2), 100, dtype=np.uint16)
    green = np.full((1, 2), 100, dtype=np.uint16)
    path = write_geotiff(str(tmp_path / "a.tif"), [blue, green, red, nir],
                          band_descriptions=["blue", "green", "red", "nir"], nodata=0)

    vegetation = masking.compute_ndvi_vegetation_mask(
        path, (0, 0, 2, 1), ["blue", "green", "red", "nir"], ndvi_threshold=0.2
    )
    assert list(vegetation[0]) == [True, False]


def test_compute_exclusion_flags_combines_vegetation(tmp_path):
    size = (2, 2)
    blue = np.full(size, 150, dtype=np.uint16)
    green = np.full(size, 120, dtype=np.uint16)
    red = np.full(size, 100, dtype=np.uint16)
    nir = np.full(size, 110, dtype=np.uint16)  # bare land: NDVI ~ 0.048, below default threshold

    red[0, 1] = 100
    nir[0, 1] = 400  # dense vegetation: NDVI = 0.6

    path = write_geotiff(
        str(tmp_path / "a.tif"), [blue, green, red, nir],
        band_descriptions=["blue", "green", "red", "nir"], nodata=0,
    )

    flags = masking.compute_exclusion_flags(
        path, None, ["blue", "green", "red", "nir"],
        use_water_mask=False, use_vegetation_mask=True, ndvi_threshold=0.2, use_omnicloudmask=False,
    )
    assert flags[0, 1] & masking.EXCLUDE_VEGETATION
    assert flags[0, 0] == 0

    flags_disabled = masking.compute_exclusion_flags(
        path, None, ["blue", "green", "red", "nir"],
        use_water_mask=False, use_vegetation_mask=False, use_omnicloudmask=False,
    )
    assert not (flags_disabled & masking.EXCLUDE_VEGETATION).any()


def test_compute_exclusion_flags_without_water_mask(tmp_path):
    green = np.array([[300, 100]], dtype=np.uint16)
    nir = np.array([[50, 300]], dtype=np.uint16)
    blue = np.full((1, 2), 100, dtype=np.uint16)
    red = np.full((1, 2), 100, dtype=np.uint16)
    path = write_geotiff(str(tmp_path / "a.tif"), [blue, green, red, nir],
                          band_descriptions=["blue", "green", "red", "nir"], nodata=0)

    flags = masking.compute_exclusion_flags(
        path, None, ["blue", "green", "red", "nir"],
        use_water_mask=False, use_omnicloudmask=False,
    )
    assert not (flags & masking.EXCLUDE_WATER).any()


def test_scene_clear_fraction_prefers_metadata(tmp_path):
    meta_path = tmp_path / "scene_metadata.json"
    meta_path.write_text(json.dumps({"properties": {"clear_percent": 87}}))
    scene = io.Scene(
        scene_id="scene", analytic_path="unused.tif", udm2_path=None,
        metadata_path=str(meta_path), acquired=None,
    )
    assert masking.scene_clear_fraction(scene) == 0.87


def test_scene_clear_fraction_falls_back_to_udm2(tmp_path):
    udm2_clear = np.ones((10, 10), dtype=np.uint8)
    udm2_clear[:3, :] = 0  # 30% not-clear
    udm2_path = write_geotiff(str(tmp_path / "udm2.tif"), [udm2_clear], nodata=None)
    scene = io.Scene(
        scene_id="scene", analytic_path="unused.tif", udm2_path=udm2_path,
        metadata_path=None, acquired=None,
    )
    assert masking.scene_clear_fraction(scene) == 0.7


def test_row_blocked_flags_match_whole_raster_masks(tmp_path):
    """compute_exclusion_flags (row blocks of 7) gives the same nodata/water/
    vegetation bits as the whole-raster mask functions; a composite's NaN
    pixel also counts as nodata."""
    import warnings as _warnings
    rng = np.random.default_rng(5)
    names = ["blue", "green", "red", "nir"]
    arrays = [rng.integers(1, 3000, (23, 17)).astype(np.uint16) for _ in range(4)]
    arrays[0][3:6, 2:9] = 0
    scene = write_geotiff(str(tmp_path / "s.tif"), arrays, band_descriptions=names, nodata=0)
    window = (0, 0, 17, 23)
    expected = ((~masking.read_nodata_valid_mask(scene, window)).astype(np.uint8) * masking.EXCLUDE_NODATA
                | masking.compute_ndwi_water_mask(scene, window, names).astype(np.uint8) * masking.EXCLUDE_WATER
                | masking.compute_ndvi_vegetation_mask(scene, window, names).astype(np.uint8) * masking.EXCLUDE_VEGETATION)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        blocked = masking.compute_exclusion_flags(scene, None, names, use_omnicloudmask=False, block_rows=7)
    np.testing.assert_array_equal(blocked, expected)
    assert (blocked & masking.EXCLUDE_WATER).any() and (blocked & masking.EXCLUDE_VEGETATION).any()

    floats = [a.astype(np.float32) for a in arrays]
    floats[0][3:6, 2:9] = -9999.0
    floats[2][10, 10] = np.nan
    composite = write_geotiff(str(tmp_path / "c.tif"), floats, band_descriptions=names, nodata=-9999.0,
                              dtype=gdal.GDT_Float32)
    composite_flags = masking.compute_composite_exclusion_flags(composite, names, block_rows=7)
    expected[10, 10] |= masking.EXCLUDE_NODATA
    np.testing.assert_array_equal(composite_flags, expected)
