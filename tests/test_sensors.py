import numpy as np
import pytest

from psnorm import sensors

from _raster_utils import write_geotiff


def test_detect_band_names_from_descriptions(tmp_path):
    arr = np.full((10, 10), 100, dtype=np.uint16)
    path = write_geotiff(
        str(tmp_path / "four_band.tif"),
        [arr, arr, arr, arr],
        band_descriptions=["blue", "green", "red", "nir"],
    )
    assert sensors.detect_band_names(path) == ["blue", "green", "red", "nir"]


def test_detect_band_names_superdove_8band_descriptions(tmp_path):
    arr = np.full((10, 10), 100, dtype=np.uint16)
    names = ["coastal blue", "blue", "green i", "green", "yellow", "red", "red edge", "nir"]
    path = write_geotiff(str(tmp_path / "eight_band.tif"), [arr] * 8, band_descriptions=names)
    assert sensors.detect_band_names(path) == [
        "coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir",
    ]


def test_detect_band_names_falls_back_to_profile_with_warning(tmp_path):
    arr = np.full((10, 10), 100, dtype=np.uint16)
    path = write_geotiff(str(tmp_path / "no_desc.tif"), [arr, arr, arr, arr])  # no descriptions
    with pytest.warns(RuntimeWarning):
        names = sensors.detect_band_names(path)
    assert names == sensors.BAND_PROFILES[4]


def test_detect_band_names_unknown_count_raises(tmp_path):
    arr = np.full((10, 10), 100, dtype=np.uint16)
    path = write_geotiff(str(tmp_path / "weird.tif"), [arr, arr, arr])  # 3 bands, no profile
    with pytest.raises(ValueError):
        sensors.detect_band_names(path)


def test_cloudmask_band_indices():
    assert sensors.cloudmask_band_indices(["blue", "green", "red", "nir"]) == (3, 2, 4)
    assert sensors.cloudmask_band_indices(["blue", "green"]) is None


def test_ndwi_band_indices():
    assert sensors.ndwi_band_indices(["blue", "green", "red", "nir"]) == (2, 4)
    assert sensors.ndwi_band_indices(["blue", "red"]) is None
