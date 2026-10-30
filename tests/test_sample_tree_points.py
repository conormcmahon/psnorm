import csv
import os
import sys

import numpy as np

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
