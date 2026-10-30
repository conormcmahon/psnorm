import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import sample_tree_points as stp  # noqa: E402

from psnorm import io, masking, model_io  # noqa: E402
from psnorm.normalize import BandModel, NormalizationModel  # noqa: E402


def _info(width, height):
    return io.RasterInfo(path="x", width=width, height=height, band_count=1, crs="",
                         geotransform=(0.0, 3.0, 0.0, 0.0, 0.0, -3.0), nodata=None)


def _make_run(tmp_path):
    root = tmp_path / "run"
    group = root / "solstice_2018"
    for sub in ("models", "candidates", "consensus", "masks"):
        (group / sub).mkdir(parents=True)
    with open(root / "composite_assignment.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "composite"])
        w.writerow(["s1", "solstice_2018"])
        w.writerow(["s2", "solstice_2018"])

    # Composite grid 6x6; scene s1 covers columns 2-5, rows 1-4 of it.
    consensus = np.zeros((6, 6), dtype=np.uint8)
    consensus[1:5, 2:6] = 1                    # 16 consensus pixels inside the scene's window
    consensus[0, 0] = 1                        # one outside it
    masking.save_flags_raster(consensus, _info(6, 6), str(group / "consensus" / "consensus_mask.tif"))
    flags = np.zeros((4, 4), dtype=np.uint8)
    flags[0, :] = masking.EXCLUDE_NODATA       # 4 nodata pixels (not counted)
    flags[1, :] = masking.EXCLUDE_UDM2         # 4 cloudy pixels (counted: unmasked)
    masking.save_flags_raster(flags, _info(4, 4), str(group / "masks" / "s1_flags.tif"))
    model_io.save_json({"ref_window": [2, 1, 4, 4], "tgt_window": [0, 0, 4, 4]},
                       str(group / "candidates" / "s1_stats.json"))
    model = NormalizationModel(
        reference_id="solstice_2018", target_id="s1", band_names=["blue", "nir"],
        bands=[
            BandModel(band_name="blue", slope=1.1, intercept=-20.0, r2=0.9, n_invariant_pixels=7,
                      rmse_before=50.0, rmse_after=10.0),
            BandModel(band_name="nir", slope=1.0, intercept=0.0, r2=float("nan"), n_invariant_pixels=3,
                      identity_fallback=True, fallback_reason="too few pixels"),
        ],
        n_consensus_pixels=8, invariance_frequency_threshold=0.3, min_observations=2,
    )
    model_io.save_model(model, str(group / "models" / "s1_model.json"))
    return str(root)


def test_normalization_fields(tmp_path):
    root = _make_run(tmp_path)
    assignment = stp.read_assignment(root)
    fields = stp.normalization_fields(root, assignment["s1"], "s1")
    assert set(fields) == {"blue", "nir"}
    blue, nir = fields["blue"], fields["nir"]
    assert blue["norm_composite"] == "solstice_2018"
    assert blue["norm_consensus_pixels_unmasked"] == 12   # 16 in the window, less the 4 nodata
    assert blue["norm_consensus_pixels_fit"] == 8
    assert (blue["norm_ransac_inliers"], blue["norm_slope"], blue["norm_intercept"]) == (7, 1.1, -20.0)
    assert (blue["norm_r2"], blue["norm_rmse_before"], blue["norm_rmse_after"]) == (0.9, 50.0, 10.0)
    assert (blue["norm_identity_fallback"], blue["norm_fallback_reason"]) == (0, "")
    assert nir["norm_r2"] == "" and nir["norm_identity_fallback"] == 1
    assert nir["norm_fallback_reason"] == "too few pixels"
    assert set(blue) == set(stp.NORM_COLUMNS)


def test_normalization_fields_without_model(tmp_path):
    root = _make_run(tmp_path)
    assert stp.normalization_fields(root, "solstice_2018", "s2") == {}   # assigned, not yet fitted
    assert stp.normalization_fields(root, None, "s3") == {}              # not in the assignment


def test_read_selected_reflectance(tmp_path):
    path = tmp_path / "preferred_imagery.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "preferred_source"])
        w.writerow(["a", "registered"])
        w.writerow(["b", "raw"])
    assert stp.read_selected_reflectance(str(path)) == {"a": "affine", "b": "original"}
    assert stp.read_selected_reflectance(str(tmp_path / "missing.csv")) == {}
    assert "selected_reflectance" in stp.COLUMNS


def test_points_values_window_matches_pixel_reads(tmp_path, monkeypatch):
    from osgeo import gdal
    path = str(tmp_path / "r.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 50, 40, 2, gdal.GDT_Int16)
    for b in (1, 2):
        arr = (np.arange(50 * 40).reshape(40, 50) * b).astype(np.int16)
        arr[5, 7] = -9999
        ds.GetRasterBand(b).WriteArray(arr)
        ds.GetRasterBand(b).SetNoDataValue(-9999)
    ds = None
    ds = gdal.Open(path)
    rng = np.random.default_rng(1)
    cols = list(rng.integers(0, 50, 30)) + [7]
    rows = list(rng.integers(0, 40, 30)) + [5]
    by_pixel = stp.points_values(ds, cols, rows)
    monkeypatch.setattr(stp, "WINDOW_READ_MIN_POINTS", 1)
    by_window = stp.points_values(ds, cols, rows)
    assert by_pixel == by_window
    assert by_window[-1] == [None, None]                 # nodata
    assert by_window[0] == [rows[0] * 50 + cols[0], 2 * (rows[0] * 50 + cols[0])]


def test_read_points_geometry_and_extra_fields(tmp_path):
    from osgeo import ogr, osr
    path = str(tmp_path / "p.gpkg")
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer("points", srs=srs, geom_type=ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn("point_id", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("class", ogr.OFTString))
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetField("point_id", 7)
    feat.SetField("class", "freq_00-05pct")
    geom = ogr.Geometry(ogr.wkbPoint)
    geom.AddPoint(-75.1, 39.9)
    feat.SetGeometry(geom)
    layer.CreateFeature(feat)
    ds = None
    (pid, lon, lat, extra), = stp.read_points(path, "point_id", ["class"])
    assert (pid, round(lon, 6), round(lat, 6), extra) == ("7", -75.1, 39.9, {"class": "freq_00-05pct"})


def test_output_columns():
    cols = stp.output_columns("point_id", ["class", "frequency"], normalization=True)
    assert cols[:4] == ["point_id", "class", "frequency", "date"]
    assert "treeid" not in cols and cols[-len(stp.NORM_COLUMNS):] == stp.NORM_COLUMNS
    assert stp.output_columns() == stp.COLUMNS


def test_normalization_fields_from_another_group_folder(tmp_path):
    # --group-dir solstice_2018=solstice_2018_rerun: the scene's fit is read
    # from the rerun folder and norm_composite names that folder.
    import shutil
    root = _make_run(tmp_path)
    shutil.move(os.path.join(root, "solstice_2018"), os.path.join(root, "solstice_2018_rerun"))
    assert stp.normalization_fields(root, "solstice_2018", "s1") == {}
    fields = stp.normalization_fields(root, "solstice_2018_rerun", "s1")
    assert fields["blue"]["norm_composite"] == "solstice_2018_rerun"
    assert fields["blue"]["norm_slope"] == 1.1


def test_with_io_retry(monkeypatch):
    import pytest
    monkeypatch.setattr(stp, "IO_RETRY_DELAY_S", 0)
    monkeypatch.delenv("PSNORM_IO_RECOVERY_CMD", raising=False)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("x.tif: Bad file descriptor")
        return "ok"
    assert stp.with_io_retry(flaky) == "ok" and len(calls) == 3
    with pytest.raises(RuntimeError, match="not a transient"):
        stp.with_io_retry(lambda: (_ for _ in ()).throw(RuntimeError("not a transient error")))


def test_normalization_fields_all_pixel_quality(tmp_path, monkeypatch):
    root = _make_run(tmp_path)
    # Not recorded in the model and no target path: blank.
    fields = stp.normalization_fields(root, "solstice_2018", "s1")
    assert fields["blue"]["norm_r2_all"] == "" and fields["blue"]["norm_rmse_after_all"] == ""
    # Recomputed (and cached) when a target path is given.
    calls = []

    def fake(group, scene_id, target_path):
        calls.append(target_path)
        return {"blue": (0.5, 300.0, 250.0)}
    monkeypatch.setattr(stp.pipeline, "consensus_fit_quality_all", fake)
    cache = str(tmp_path / "cache")
    fields = stp.normalization_fields(root, "solstice_2018", "s1", "/x/s1.tif", cache)
    assert (fields["blue"]["norm_r2_all"], fields["blue"]["norm_rmse_before_all"],
            fields["blue"]["norm_rmse_after_all"]) == (0.5, 300.0, 250.0)
    assert fields["nir"]["norm_r2_all"] == ""
    fields = stp.normalization_fields(root, "solstice_2018", "s1", "/x/s1.tif", cache)
    assert fields["blue"]["norm_r2_all"] == 0.5 and calls == ["/x/s1.tif"]   # second call from the cache
    # Taken from the model where it records them.
    model = model_io.load_model(os.path.join(root, "solstice_2018", "models", "s1_model.json"))
    for band in model.bands:
        band.r2_all, band.rmse_before_all, band.rmse_after_all = 0.7, 40.0, 30.0
    model_io.save_model(model, os.path.join(root, "solstice_2018", "models", "s1_model.json"))
    fields = stp.normalization_fields(root, "solstice_2018", "s1")
    assert fields["nir"]["norm_r2_all"] == 0.7 and len(calls) == 1


def test_points_array_masks_nodata_only_in_the_first_rows(tmp_path):
    from osgeo import gdal
    path = str(tmp_path / "r.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 20, 10, 1, gdal.GDT_Int16)
    arr = np.arange(200, dtype=np.int16).reshape(10, 20)
    arr[2, 3] = -9999
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(-9999)
    ds = None
    cols, rows = [3, 4, 3], [2, 2, 2]          # nodata, data, nodata again
    assert np.isnan(stp.points_array(path, cols, rows)[[0, 2], 0]).all()
    kept = stp.points_array(path, cols, rows, n_masked=1)[:, 0]
    assert np.isnan(kept[0]) and kept[1] == 44 and kept[2] == -9999   # raw value beyond n_masked
    # the window path (many points) agrees with the per-pixel path
    many_c = list(np.tile(np.arange(20), 10)) + cols
    many_r = list(np.repeat(np.arange(10), 20)) + rows
    window = stp.points_array(path, many_c, many_r)
    assert np.array_equal(window[:200, 0][~np.isnan(window[:200, 0])],
                          arr.ravel()[arr.ravel() != -9999].astype(float))


def _fake_samples():
    """Two scenes' SceneSamples over three points (no rasters needed)."""
    points = [("11", -75.1, 39.9, {"genus": "Acer"}), ("5", -75.2, 39.8, {"genus": "Quercus"}),
              ("7", -75.3, 39.7, {"genus": "Acer"})]
    samples = []
    for sid, acquired, idx in (("20200602_150000_b", "2020-06-02T15:00:00", [0, 2]),
                               ("20200601_150000_a", "2020-06-01T15:00:00", [0, 1, 2])):
        n = len(idx)
        vals = np.arange(n * 2, dtype=float).reshape(n, 2) * 100 + 1000
        vals[0, 1] = np.nan
        scene_fields = {
            "date": acquired[:10], "acquired": acquired, "scene_id": sid, "product": "p8b",
            "selected_reflectance": "affine", "ransac_inliers_affine": "120", "ransac_inliers_tps": "",
            "status_affine": "ok", "status_tps": "not yet run", "affine_scale_x": "1.001", "affine_scale_y": "",
            "affine_rotation_deg": "0.1", "affine_shear_deg": "", "affine_offset_x_px": "0.5",
            "affine_offset_y_px": "-0.25", "affine_offset_x_m": 1.5, "affine_offset_y_m": -0.75,
        }
        metrics = [{k: "" for k in stp.COLUMNS if k.startswith(("r2_", "rmse_"))} for _ in range(2)]
        metrics[0]["r2_original_all"] = "0.5"
        norm = {"blue": {c: "" for c in stp.NORM_COLUMNS}, "nir": {c: "" for c in stp.NORM_COLUMNS}}
        norm["blue"].update(norm_composite="solstice_2020", norm_slope=1.05, norm_intercept=-10.0,
                            norm_ransac_inliers=900, norm_identity_fallback=0, norm_r2_all=0.8)
        samples.append(stp.SceneSample(points, np.array(idx), np.array([3, 4, 5][:n]), np.array([6, 7, 8][:n]),
                                       ["blue", "nir"], {"original": vals, "affine": vals + 1, "tps": None},
                                       np.array([1.0] + [np.nan] * (n - 1)), scene_fields, metrics, norm))
    return points, samples


def test_parquet_output_matches_the_csv_rows(tmp_path):
    import types
    import pandas as pd
    import point_samples_parquet as psp
    points, samples = _fake_samples()
    points_csv = tmp_path / "points.csv"
    with open(points_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TREEID", "GENUS", "LON", "LAT"])
        for pid, lon, lat, extra in points:
            w.writerow([pid, extra["genus"], lon, lat])

    class Ctx:
        def __init__(self):
            self.points = stp.read_points(str(points_csv), "treeid", ["genus"])
            self.by_id = {s.scene_fields["scene_id"]: s for s in samples}

        def sample_scene(self, task):
            return self.by_id[task["scene_id"]]

    args = types.SimpleNamespace(points=str(points_csv), id_field="treeid", partition_by="genus,treeid",
                                 staging_dir=None, output=str(tmp_path / "out"), buckets=2, flush_rows=5, shard=None)
    tasks = [{"scene_id": s.scene_fields["scene_id"]} for s in samples]
    psp.sample_to_staging(Ctx(), tasks, args, ["genus"])
    psp.sample_to_staging(Ctx(), tasks, args, ["genus"])     # a rerun stages nothing twice
    psp.finalize(args, ["genus"])
    got = pd.read_parquet(args.output)
    assert sorted(os.listdir(os.path.join(args.output, "genus=Acer"))) == ["treeid=11", "treeid=7"]

    expected = pd.DataFrame([r for s in samples for r in s.csv_records("treeid")])
    assert len(got) == len(expected) == 10
    got = got.assign(treeid=got.treeid.astype(int).astype(str), genus=got.genus.astype(str),
                     date=got.date.astype(str), acquired=got.acquired.dt.strftime("%Y-%m-%dT%H:%M:%S"))
    key = ["treeid", "scene_id", "band"]
    m = expected.merge(got, on=key, suffixes=("_csv", "_pq"))
    assert len(m) == 10
    for col in expected.columns:
        if col in key:
            continue
        a = m[col + "_csv"].replace("", np.nan)
        b = m[col + "_pq"]
        a_num, b_num = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        if a.notna().any() and a_num.notna().sum() == a.notna().sum():      # a numeric column
            assert np.allclose(a_num.astype(float), b_num.astype(float), equal_nan=True), col
        else:
            assert (a.fillna("").astype(str) == b.fillna("").astype(str)).all(), col
    # each tree's rows are in time order
    one = pd.read_parquet(os.path.join(args.output, "genus=Acer", "treeid=11"))
    assert one.acquired.is_monotonic_increasing and list(one.year.unique()) == [2020]


def test_consensus_reference_gives_the_fit_pixels_of_pipeline_recompute(tmp_path):
    # A small composite-reference run, then the sampler's one-read-per-group
    # path must select the same pixels and give the same metrics as
    # pipeline.consensus_fit_quality_all.
    from osgeo import gdal
    from psnorm import pipeline
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _raster_utils import write_geotiff
    from test_pipeline import _write_varying_scene
    input_dir, output_dir, size = tmp_path / "input", tmp_path / "output", 60
    input_dir.mkdir()
    rng = np.random.default_rng(3)
    ref = {b: rng.uniform(lo, hi, (size, size)) for b, (lo, hi) in
           {"blue": (200, 3000), "green": (100, 1500), "red": (150, 2000), "nir": (300, 3500)}.items()}
    bands, names = [], []
    for name in ["blue", "green", "red", "nir"]:
        for stat in ("median", "mean"):
            bands.append(ref[name].astype(np.float32))
            names.append(f"{name}_{stat}")
    composite = write_geotiff(str(tmp_path / "solstice_2020.tif"), bands, band_descriptions=names,
                              nodata=-9999.0, dtype=gdal.GDT_Float32)
    noise = np.random.default_rng(1)
    scene_arrays = [np.clip(ref[b] / 1.1 + noise.normal(0, 30, (size, size)), 1, None).astype(np.uint16)
                    for b in ("blue", "green", "red", "nir")]
    _write_varying_scene(input_dir, "20200601_120000_a", scene_arrays)
    _write_varying_scene(input_dir, "20200602_120000_b", [a + 5 for a in scene_arrays])
    pipeline.run_pipeline(str(input_dir), str(output_dir), reference_composite=composite,
                          composite_cache_dir=str(tmp_path / "cache"), use_omnicloudmask=False,
                          use_water_mask=False, use_vegetation_mask=False, min_overlap_pixels=100, min_fit_pixels=2,
                          invariance_frequency_threshold=0.1, min_observations=1, device="cpu", workers=None)
    scene = next(s for s in io.discover_scenes(str(input_dir)) if s.scene_id == "20200601_120000_a")
    expected = pipeline.consensus_fit_quality_all(str(output_dir), scene.scene_id, scene.analytic_path)
    model = model_io.load_model(pipeline._model_path(str(output_dir), scene.scene_id))
    reference = stp.ConsensusReference(str(output_dir), model.reference_id)
    stats = model_io.load_json(os.path.join(output_dir, "candidates", f"{scene.scene_id}_stats.json"))
    positions, cols, rows = reference.scene_pixels(stats["ref_window"], stats["tgt_window"],
                                                   os.path.join(output_dir, "masks", f"{scene.scene_id}_search_mask.tif"))
    assert positions.size == model.n_consensus_pixels > 0
    values = stp.points_array(scene.analytic_path, cols, rows, n_masked=0)
    ctx = stp.SceneContext.__new__(stp.SceneContext)
    got = ctx._quality_from_plan({"model": model, "reference": reference, "positions": positions}, values,
                                 ["blue", "green", "red", "nir"])
    assert set(got) == set(expected)
    for band in expected:
        assert got[band] == pytest.approx(expected[band])


def test_parquet_slim_layout_joins_back_to_the_full_rows(tmp_path):
    import types
    import pandas as pd
    import point_samples_parquet as psp
    points, samples = _fake_samples()
    points_csv = tmp_path / "points.csv"
    with open(points_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TREEID", "GENUS", "LON", "LAT"])
        for pid, lon, lat, extra in points:
            w.writerow([pid, extra["genus"], lon, lat])

    class Ctx:
        points = stp.read_points(str(points_csv), "treeid", ["genus"])

        def sample_scene(self, task):
            return {s.scene_fields["scene_id"]: s for s in samples}[task["scene_id"]]

    staging = str(tmp_path / "staging")
    common = dict(points=str(points_csv), id_field="treeid", partition_by="genus,treeid", staging_dir=staging,
                  buckets=2, flush_rows=5, shard=None)
    full = types.SimpleNamespace(output=str(tmp_path / "full"), layout="full", **common)
    slim = types.SimpleNamespace(output=str(tmp_path / "slim"), layout="slim", **common)
    psp.sample_to_staging(Ctx(), [{"scene_id": s.scene_fields["scene_id"]} for s in samples], full, ["genus"])
    psp.finalize(full, ["genus"])
    psp.finalize(slim, ["genus"])          # the same staging, finalized again in the slim layout

    trees = pd.read_parquet(slim.output)
    assert set(trees.columns) == {"treeid", "genus", *psp.TREE_COLUMNS}
    scene_bands = pd.read_parquet(os.path.join(slim.output, psp.SCENE_TABLE_NAME))
    assert len(scene_bands) == 4 and not scene_bands.duplicated(["scene_id", "band"]).any()
    assert "norm_slope" in scene_bands and "reflectance_original" not in scene_bands
    joined = trees.merge(scene_bands.drop(columns=[c for c in psp.TREE_COLUMNS if c in scene_bands
                                                   and c not in ("scene_id", "band")]), on=["scene_id", "band"])
    expected = pd.read_parquet(full.output)
    key = ["treeid", "scene_id", "band"]
    joined = joined.sort_values(key).reset_index(drop=True)
    expected = expected.sort_values(key).reset_index(drop=True)
    assert sorted(joined.columns) == sorted(expected.columns)
    pd.testing.assert_frame_equal(joined[expected.columns], expected, check_categorical=False)
