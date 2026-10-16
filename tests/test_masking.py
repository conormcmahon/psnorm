import json

import numpy as np

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
