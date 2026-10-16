import numpy as np

from psnorm import io, metrics

from _raster_utils import write_geotiff


def test_agreement_between_identical_images_is_perfect(tmp_path):
    rng = np.random.default_rng(0)
    band = rng.integers(1, 2000, size=(20, 20)).astype(np.uint16)
    path_a = write_geotiff(str(tmp_path / "a.tif"), [band, band], band_descriptions=["red", "nir"])
    path_b = write_geotiff(str(tmp_path / "b.tif"), [band, band], band_descriptions=["red", "nir"])

    window = (0, 0, 20, 20)
    mask = np.ones((20, 20), dtype=bool)
    result = metrics.agreement_between(path_a, window, ["red", "nir"], path_b, window, ["red", "nir"], mask)

    assert result.n_pixels == 400
    for band_result in result.bands:
        assert band_result.r2 == 1.0
        assert band_result.rmse == 0.0
    assert result.pooled_r2 == 1.0
    assert result.pooled_rmse == 0.0


def test_agreement_between_detects_known_bias(tmp_path):
    rng = np.random.default_rng(1)
    band_a = rng.integers(100, 1000, size=(30, 30)).astype(np.uint16)
    offset = 50
    band_b = (band_a.astype(np.int32) + offset).astype(np.uint16)

    path_a = write_geotiff(str(tmp_path / "a.tif"), [band_a], band_descriptions=["red"])
    path_b = write_geotiff(str(tmp_path / "b.tif"), [band_b], band_descriptions=["red"])

    window = (0, 0, 30, 30)
    mask = np.ones((30, 30), dtype=bool)
    result = metrics.agreement_between(path_a, window, ["red"], path_b, window, ["red"], mask)

    assert result.bands[0].r2 > 0.999  # perfectly linearly related (just offset)
    assert result.bands[0].rmse == 50.0  # constant difference -> RMSE == offset


def test_agreement_between_only_uses_masked_pixels(tmp_path):
    band_a = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    band_b = np.array([[10, 999], [30, 999]], dtype=np.uint16)  # disagreement only where masked out

    path_a = write_geotiff(str(tmp_path / "a.tif"), [band_a], band_descriptions=["red"])
    path_b = write_geotiff(str(tmp_path / "b.tif"), [band_b], band_descriptions=["red"])

    window = (0, 0, 2, 2)
    mask = np.array([[True, False], [True, False]])
    result = metrics.agreement_between(path_a, window, ["red"], path_b, window, ["red"], mask)

    assert result.n_pixels == 2
    assert result.bands[0].rmse == 0.0  # the two masked-in columns agree exactly
