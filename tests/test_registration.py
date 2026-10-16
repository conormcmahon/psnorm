"""Tests for registration.py: parameter defaults/plumbing, band and
warp-strategy selection, UDM2 path resolution, and output writing.
planet_georeg_opencv calls are monkeypatched where the point is psnorm's
own logic; write_registered runs for real on a small synthetic raster.
"""

import argparse
import functools
import inspect
import os
from types import SimpleNamespace

import numpy as np
import planet_georeg_opencv as pgo
from planet_georeg_opencv import tps_warp
import pytest
from osgeo import gdal

from psnorm import composite_reference, io, registration, sensors

gdal.UseExceptions()


def _raster_info(width=10, height=10, band_count=4, crs="EPSG:32618",
                  geotransform=(500000.0, 3.0, 0.0, 4400000.0, 0.0, -3.0)):
    return io.RasterInfo(path="x.tif", width=width, height=height, band_count=band_count,
                          crs=crs, geotransform=geotransform, nodata=None)


def _patch_register_local(monkeypatch, fake):
    """Stand in for the library's local registration with `fake`, written
    as a register_local replacement: it is called once per warp model."""
    def fake_models(moving, fixed, band_map, warp_models, *, on_warped_band=None, **kwargs):
        return {
            model: fake(moving, fixed, band_map, warp_model=model,
                        on_warped_band=functools.partial(on_warped_band, model) if on_warped_band else None, **kwargs)
            for model in warp_models
        }
    monkeypatch.setattr(registration.pgo, "register_local_models", fake_models)


def _ok_result():
    return SimpleNamespace(status="ok", registered=None, metrics={}, message="", remap=None, diagnostics={})


@pytest.fixture
def plain_scene_env(monkeypatch):
    """Grid-aligned target + plain-scene reference, both 4-band, no UDM2."""
    monkeypatch.setattr(io, "get_raster_info", lambda p: _raster_info())
    monkeypatch.setattr(sensors, "detect_band_names", lambda p: ["blue", "green", "red", "nir"])
    monkeypatch.setattr(composite_reference, "is_composite", lambda p: False)
    monkeypatch.setattr(registration, "udm2_path_for", lambda p: None)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

def test_library_parameter_defaults_match_planet_georeg_opencv():
    """Every RegistrationParams field passed to the library must default to
    the library's own default (affine warp and the 500 m buffer are the
    only intentional psnorm-side differences)."""
    library_defaults = {
        name: p.default for name, p in inspect.signature(pgo.register_local).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    kwargs = registration.RegistrationParams().register_local_kwargs()
    differing = {k: (v, library_defaults[k]) for k, v in kwargs.items() if v != library_defaults[k]}
    assert differing == {"warp_model": ("affine", "tps"), "fixed_buffer_m": (500.0, 0.0),
                         "sift_tile_size": (1024, None)}


def test_params_reject_unknown_warp_model():
    with pytest.raises(ValueError):
        registration.RegistrationParams(warp_model="spline")


def test_cli_arguments_round_trip():
    parser = argparse.ArgumentParser()
    registration.add_registration_arguments(parser)
    defaults = registration.registration_params_from_args(parser.parse_args([]))
    assert defaults == registration.RegistrationParams()

    args = parser.parse_args([
        "--registration-warp-model", "tps", "--registration-ransac-reproj-threshold", "3.5",
        "--registration-use-udm2-mask", "false", "--registration-match-bands", "all",
        "--registration-tps-max-control-points", "none", "--registration-max-sift-dimension", "2500",
        "--registration-reference-keypoint-cache", "false", "--registration-reference-keypoint-scales", "1,0.6",
        "--registration-sift-tile-size", "none",
        "--registration-reference-cache-dir", "/fast/Cache Dir", "--registration-reference-cache-compression", "NONE",
    ])
    params = registration.registration_params_from_args(args)
    assert params.warp_model == "tps"
    assert params.ransac_reproj_threshold == 3.5
    assert params.use_udm2_mask is False
    assert params.match_bands is None
    assert params.tps_max_control_points is None
    assert params.max_sift_dimension == 2500 and isinstance(params.max_sift_dimension, int)
    assert params.reference_keypoint_cache is False
    assert params.reference_keypoint_scales == (1.0, 0.6)
    assert params.sift_tile_size is None and defaults.sift_tile_size == 1024
    assert params.reference_cache_dir == "/fast/Cache Dir" and defaults.reference_cache_dir is None
    assert params.reference_cache_compression == "NONE"
    with pytest.raises(ValueError):
        registration.RegistrationParams(reference_cache_compression="zip")


def test_udm2_path_for(tmp_path):
    for analytic in ("20210902_145552_70_2428_3B_AnalyticMS_SR_8b_harmonized_clip.tif",
                     "20180501_151332_100c_3B_AnalyticMS_SR_harmonized_clip.tif"):
        prefix = analytic.split("_3B_")[0]
        (tmp_path / analytic).touch()
        assert registration.udm2_path_for(str(tmp_path / analytic)) is None  # sidecar missing
        (tmp_path / f"{prefix}_3B_udm2_clip.tif").touch()
        assert registration.udm2_path_for(str(tmp_path / analytic)) == str(tmp_path / f"{prefix}_3B_udm2_clip.tif")


# --------------------------------------------------------------------------
# Strategy / band selection
# --------------------------------------------------------------------------

def test_grid_aligned_uses_register_local_with_params(plain_scene_env, monkeypatch, tmp_path):
    calls = {}

    def fake_register_local(moving, fixed, band_map, **kwargs):
        calls.update(band_map=band_map, kwargs=kwargs)
        return _ok_result()

    _patch_register_local(monkeypatch, fake_register_local)
    monkeypatch.setattr(registration.pgo, "register_to_grid", lambda *a, **k: pytest.fail("not grid-aligned path"))

    params = registration.RegistrationParams(ransac_reproj_threshold=3.0, buffer_m=250.0)
    out = registration.register_scene(str(tmp_path / "t.tif"), str(tmp_path / "r.tif"), params)

    assert out.strategy == "local"
    assert calls["band_map"] == [0, 1, 2, 3]
    assert calls["kwargs"]["ransac_reproj_threshold"] == 3.0
    assert calls["kwargs"]["fixed_buffer_m"] == 250.0
    assert calls["kwargs"]["warp_model"] == "affine"
    assert calls["kwargs"]["return_registered"] is False


def test_not_grid_aligned_uses_register_to_grid(plain_scene_env, monkeypatch, tmp_path):
    target, reference = str(tmp_path / "t.tif"), str(tmp_path / "r.tif")
    infos = {target: _raster_info(), reference: _raster_info(geotransform=(500000.0, 0.6, 0.0, 4400000.0, 0.0, -0.6))}
    monkeypatch.setattr(io, "get_raster_info", lambda p: infos[p])
    _patch_register_local(monkeypatch, lambda *a, **k: pytest.fail("grid-aligned path"))
    monkeypatch.setattr(registration.pgo, "register_to_grid", lambda *a, **k: _ok_result())

    assert registration.register_scene(target, reference).strategy == "grid"


def test_composite_reference_extracts_only_match_bands(monkeypatch, tmp_path):
    eight = ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"]
    monkeypatch.setattr(io, "get_raster_info", lambda p: _raster_info(band_count=8))
    monkeypatch.setattr(sensors, "detect_band_names", lambda p: eight)
    monkeypatch.setattr(composite_reference, "is_composite", lambda p: True)
    monkeypatch.setattr(registration, "udm2_path_for", lambda p: None)

    calls = {}
    sentinel = object()

    def fake_extract(path, target_info, band_names, *, stat, buffer_m, cache_dir, cache_compression):
        calls["extract"] = (band_names, stat, buffer_m)
        calls["cache"] = (cache_dir, cache_compression)
        return sentinel

    def fake_register_local(moving, fixed, band_map, **kwargs):
        calls.update(fixed=fixed, band_map=band_map, fixed_keypoints=kwargs["fixed_keypoints"])
        return _ok_result()

    keypoints = object()

    def fake_keypoints(reference_path, band_names, target_info, reference_window, params):
        calls["keypoint_bands"] = band_names
        return keypoints, 0.7

    monkeypatch.setattr(composite_reference, "extract_reference_window", fake_extract)
    monkeypatch.setattr(registration, "_reference_keypoints_for", fake_keypoints)
    _patch_register_local(monkeypatch, fake_register_local)

    out = registration.register_scene(str(tmp_path / "t.tif"), str(tmp_path / "solstice_2024.tif"))

    assert calls["extract"] == (["blue", "green", "red", "nir"], "median", 500.0)
    assert calls["fixed"] is sentinel
    # Each 8-band target band -> its position in the 4 extracted bands, or -1.
    assert calls["band_map"] == [-1, 0, -1, 1, -1, 2, -1, 3]
    # The composite's stored keypoints, for the same bands in the same order.
    assert calls["keypoint_bands"] == ["blue", "green", "red", "nir"]
    assert calls["fixed_keypoints"] is keypoints and out.reference_keypoint_scale == 0.7

    assert calls["cache"] == (None, "LZW")

    # Switched off: the library detects the composite's keypoints itself.
    # The derived files' location and compression follow the parameters.
    out = registration.register_scene(
        str(tmp_path / "t.tif"), str(tmp_path / "solstice_2024.tif"),
        registration.RegistrationParams(reference_keypoint_cache=False, reference_cache_dir="/fast/cache",
                                        reference_cache_compression="NONE"))
    assert calls["fixed_keypoints"] is None and out.reference_keypoint_scale is None
    assert calls["cache"] == ("/fast/cache", "NONE")


def test_reference_keypoint_scale_is_the_stored_one_nearest_the_scene_scale(monkeypatch):
    asked = []
    monkeypatch.setattr(composite_reference, "ensure_reference_keypoints",
                        lambda path, names, **kw: asked.append(kw["detection_scale"]) or "dir")
    monkeypatch.setattr(composite_reference, "load_reference_keypoints", lambda d, bounds: "keypoints")
    window = SimpleNamespace(rio=SimpleNamespace(bounds=lambda: (0, 0, 1, 1)))
    params = registration.RegistrationParams()  # max_sift_dimension 4000, scales 1.0 / 0.7 / 0.5
    for larger_side, expected in [(3000, 1.0), (4500, 1.0), (5200, 0.7), (6500, 0.7), (7000, 0.5), (9990, 0.5)]:
        info = _raster_info(width=2000, height=larger_side)
        assert registration._reference_keypoints_for("c.tif", ["blue", "nir"], info, window, params) == \
            ("keypoints", expected)
    assert asked == [1.0, 1.0, 0.7, 0.7, 0.5, 0.5]


def test_match_bands_restricts_matching_bands(plain_scene_env, monkeypatch, tmp_path):
    calls = {}
    _patch_register_local(monkeypatch, lambda m, f, band_map, **k: calls.update(band_map=band_map) or _ok_result())
    params = registration.RegistrationParams(match_bands=("blue", "nir"))
    registration.register_scene(str(tmp_path / "t.tif"), str(tmp_path / "r.tif"), params)
    assert calls["band_map"] == [0, -1, -1, 3]


def test_udm2_clear_mask_is_passed_to_library(plain_scene_env, monkeypatch, tmp_path):
    clear = np.zeros((10, 10), dtype=bool)
    clear[:, 5:] = True
    monkeypatch.setattr(registration, "_clear_mask", lambda path, info: clear)
    calls = {}
    _patch_register_local(monkeypatch, lambda m, f, b, **k: calls.update(k) or _ok_result())

    out = registration.register_scene(str(tmp_path / "t.tif"), str(tmp_path / "r.tif"),
                                      target_udm2_path=str(tmp_path / "t_udm2.tif"))
    assert calls["moving_valid_mask"] is clear
    assert out.target_mask_used and out.target_clear_fraction == 0.5

    out = registration.register_scene(str(tmp_path / "t.tif"), str(tmp_path / "r.tif"),
                                      registration.RegistrationParams(use_udm2_mask=False),
                                      target_udm2_path=str(tmp_path / "t_udm2.tif"))
    assert calls["moving_valid_mask"] is None and not out.target_mask_used


def test_insufficient_shared_bands_raises(plain_scene_env, monkeypatch, tmp_path):
    monkeypatch.setattr(sensors, "detect_band_names", lambda p: ["swir1"] if "t.tif" in p else ["blue", "green"])
    with pytest.raises(registration.RegistrationFailed):
        registration.register_scene(str(tmp_path / "t.tif"), str(tmp_path / "r.tif"))


def test_register_to_reference_raises_on_failed_status(plain_scene_env, monkeypatch, tmp_path):
    _patch_register_local(monkeypatch, lambda *a, **k: SimpleNamespace(status="insufficient_matches_coarse", registered=None,
                                                          message="too few matches", metrics={}, diagnostics={}))
    with pytest.raises(registration.RegistrationFailed):
        registration.register_to_reference(str(tmp_path / "t.tif"), str(tmp_path / "r.tif"))


# --------------------------------------------------------------------------
# Output writing
# --------------------------------------------------------------------------

def _write_synthetic_scene(path):
    """A 4-band uint16 scene with a 3-column nodata margin; returns its pixels."""
    ds = gdal.GetDriverByName("GTiff").Create(path, 40, 30, 4, gdal.GDT_UInt16)
    ds.SetGeoTransform((500000.0, 3.0, 0.0, 4400000.0, 0.0, -3.0))
    ds.SetProjection('LOCAL_CS["test"]')
    rng = np.random.default_rng(0)
    original = rng.integers(100, 4000, size=(4, 30, 40)).astype(np.uint16)
    original[:, :, :3] = 0  # nodata margin
    for i, name in enumerate(["blue", "green", "red", "nir"], start=1):
        band = ds.GetRasterBand(i)
        band.WriteArray(original[i - 1])
        band.SetNoDataValue(0)
        band.SetDescription(name)
    ds = None
    return original


def test_register_scene_streams_output_and_matches_write_registered(plain_scene_env, monkeypatch, tmp_path):
    """With output_path, the bands the library hands over while registering
    are written directly; the file must equal what write_registered makes
    from the same result afterwards."""
    import cv2
    import rioxarray as rxr

    path = str(tmp_path / "scene_3B_AnalyticMS_SR_harmonized_clip.tif")
    _write_synthetic_scene(path)
    remap = tps_warp.remap_maps(np.full((30, 40), 2.0, dtype=np.float32), np.zeros((30, 40), dtype=np.float32))

    def fake_register_local(moving, fixed, band_map, *, on_warped_band=None, return_registered, **kwargs):
        assert return_registered is False
        if on_warped_band is not None:
            image = rxr.open_rasterio(moving, masked=True, cache=False).squeeze()
            for band, warped in enumerate(tps_warp.iter_remapped_bands(image, *remap, cv2.INTER_LINEAR)):
                on_warped_band(band, warped)
        return SimpleNamespace(status="ok", registered=None, metrics={}, message="", remap=remap, diagnostics={})

    _patch_register_local(monkeypatch, fake_register_local)
    monkeypatch.setattr(io, "get_raster_info", lambda p: _raster_info(width=40, height=30))

    streamed_path = str(tmp_path / "streamed.tif")
    scene = registration.register_scene(path, str(tmp_path / "r.tif"), output_path=streamed_path)
    assert scene.output_path == streamed_path

    plain = registration.register_scene(path, str(tmp_path / "r.tif"))
    assert plain.output_path is None
    later_path = registration.write_registered(plain, path, str(tmp_path / "later.tif"))

    streamed, later = gdal.Open(streamed_path), gdal.Open(later_path)
    np.testing.assert_array_equal(streamed.ReadAsArray(), later.ReadAsArray())
    assert streamed.GetMetadata("IMAGE_STRUCTURE")["INTERLEAVE"] == "BAND"
    for i in range(1, 5):
        assert streamed.GetRasterBand(i).GetDescription() == later.GetRasterBand(i).GetDescription() != ""
        assert streamed.GetRasterBand(i).GetNoDataValue() == 0


def test_register_scene_models_writes_each_model_from_one_library_call(plain_scene_env, monkeypatch, tmp_path):
    """Two warp models: the library is called once for both, each model's
    bands go to that model's own file, and a model without a path is
    registered but not written."""
    path = str(tmp_path / "scene_3B_AnalyticMS_SR_harmonized_clip.tif")
    original = _write_synthetic_scene(path)
    calls = []

    def fake_models(moving, fixed, band_map, warp_models, *, on_warped_band=None, **kwargs):
        calls.append((warp_models, kwargs))
        for model in warp_models:
            for band in range(4):
                # Recognisable per model: affine = the scene, tps = the scene + 1.
                on_warped_band(model, band, original[band].astype(np.float32) + (model == "tps"))
        return {m: SimpleNamespace(status="ok", registered=None, metrics={}, message="", remap=None, diagnostics={})
                for m in warp_models}

    monkeypatch.setattr(registration.pgo, "register_local_models", fake_models)
    monkeypatch.setattr(io, "get_raster_info", lambda p: _raster_info(width=40, height=30))

    outputs = {"affine": str(tmp_path / "a" / "out.tif"), "tps": str(tmp_path / "t" / "out.tif")}
    for out in outputs.values():
        os.makedirs(os.path.dirname(out))
    scenes = registration.register_scene_models(path, str(tmp_path / "r.tif"), outputs)

    assert len(calls) == 1 and calls[0][0] == ("affine", "tps")
    assert "warp_model" not in calls[0][1] and calls[0][1]["return_registered"] is False
    assert {m: s.output_path for m, s in scenes.items()} == outputs
    np.testing.assert_array_equal(gdal.Open(outputs["affine"]).ReadAsArray(), original)
    np.testing.assert_array_equal(gdal.Open(outputs["tps"]).ReadAsArray(), original + 1)

    only_affine = registration.register_scene_models(
        path, str(tmp_path / "r.tif"), {"affine": str(tmp_path / "again.tif"), "tps": None})
    assert only_affine["affine"].output_path and only_affine["tps"].output_path is None
    assert only_affine["tps"].result.status == "ok"


def test_register_scene_uses_params_warp_model(plain_scene_env, monkeypatch, tmp_path):
    seen = []
    _patch_register_local(monkeypatch, lambda m, f, band_map, **k: seen.append(k["warp_model"]) or _ok_result())
    registration.register_scene(str(tmp_path / "t.tif"), str(tmp_path / "r.tif"))
    registration.register_scene(str(tmp_path / "t.tif"), str(tmp_path / "r.tif"),
                                registration.RegistrationParams(warp_model="tps"))
    assert seen == ["affine", "tps"]


def test_register_scene_writes_nothing_when_registration_fails(plain_scene_env, monkeypatch, tmp_path):
    _patch_register_local(monkeypatch, lambda *a, **k: SimpleNamespace(status="insufficient_matches_coarse", registered=None,
                                                          message="too few matches", metrics={}, diagnostics={}))
    out_path = tmp_path / "out.tif"
    scene = registration.register_scene(str(tmp_path / "t.tif"), str(tmp_path / "r.tif"), output_path=str(out_path))
    assert scene.result.status == "insufficient_matches_coarse"
    assert scene.output_path is None and not out_path.exists()


def test_register_scene_removes_partial_output_if_registration_raises(plain_scene_env, monkeypatch, tmp_path):
    path = str(tmp_path / "scene_3B_AnalyticMS_SR_harmonized_clip.tif")
    _write_synthetic_scene(path)

    def failing_register_local(moving, fixed, band_map, *, on_warped_band=None, **kwargs):
        on_warped_band(0, np.zeros((30, 40), dtype=np.float32))
        raise MemoryError("simulated failure part-way through the warp")

    _patch_register_local(monkeypatch, failing_register_local)
    out_path = tmp_path / "out.tif"
    with pytest.raises(MemoryError):
        registration.register_scene(path, str(tmp_path / "r.tif"), output_path=str(out_path))
    assert not out_path.exists()


def test_write_registered_preserves_dtype_nodata_and_descriptions(tmp_path):
    path = str(tmp_path / "scene_3B_AnalyticMS_SR_harmonized_clip.tif")
    original = _write_synthetic_scene(path)

    # A whole-pixel shift of +2 columns: output(col) = input(col + 2).
    ddx = np.full((30, 40), 2.0, dtype=np.float32)
    ddy = np.zeros((30, 40), dtype=np.float32)
    reg = registration.SceneRegistration(
        result=SimpleNamespace(status="ok", remap=tps_warp.remap_maps(ddx, ddy), registered=None),
        strategy="local", band_names=["blue", "green", "red", "nir"], band_map=[0, 1, 2, 3],
        target_mask_used=False, target_clear_fraction=None,
    )
    out_path = registration.write_registered(reg, path, str(tmp_path / "out.tif"))

    out = gdal.Open(out_path)
    assert out.GetRasterBand(1).DataType == gdal.GDT_UInt16
    assert out.GetGeoTransform() == (500000.0, 3.0, 0.0, 4400000.0, 0.0, -3.0)
    assert [out.GetRasterBand(i).GetDescription() for i in range(1, 5)] == ["blue", "green", "red", "nir"]
    assert all(out.GetRasterBand(i).GetNoDataValue() == 0 for i in range(1, 5))
    values = out.ReadAsArray()
    np.testing.assert_array_equal(values[:, :, 1:38], original[:, :, 3:40])  # shifted content
    assert (values[:, :, 0] == 0).all()  # sampled from the nodata margin (input cols 0-2) -> nodata
    assert (values[:, :, 38:] == 0).all()  # sampled from beyond the edge -> nodata
