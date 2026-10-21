"""scripts/register_to_solstice_composites.py: the per-model bookkeeping
around registration (the registration itself is replaced by a stand-in)."""

import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import register_to_solstice_composites as batch  # noqa: E402
from planet_georeg_opencv import tps_warp  # noqa: E402
from psnorm import preferred_imagery, registration  # noqa: E402


def _task(tmp_path, models=("affine", "tps")):
    return {
        "scene_id": "20240101_120000_abcd", "product": "philadelphia4b", "acquired": "2024-01-01T12:00:00",
        "analytic_path": "in/20240101_120000_abcd_3B_AnalyticMS_SR_harmonized_clip.tif", "udm2_path": None,
        "reference_path": "solstice_2024.tif", "reference_year": 2024, "days_from_solstice": -171.0,
        "width": 100, "height": 50, "band_count": 4, "megapixels": 0.005, "input_bytes": 1234,
        "warp_outputs": {m: str(tmp_path / m / "philadelphia4b" / "p" / "scene.tif") for m in models},
    }


def _band_metrics(improved=True):
    """Whole-image R^2/RMSE before and after for every band of a 4-band
    scene, improved (or worsened) by registration."""
    sign = 1 if improved else -1
    metrics = {}
    for b in range(4):
        for subset in ("all", "inliers"):
            metrics[f"r2_band{b}_before_{subset}"] = 0.80
            metrics[f"r2_band{b}_after_{subset}"] = 0.80 + sign * 0.05
            metrics[f"rmse_band{b}_before_{subset}"] = 300.0
            metrics[f"rmse_band{b}_after_{subset}"] = 300.0 - sign * 20.0
    return metrics


def _scene(status="ok", dx=0.25, improved=True, guided=1500, output_path=None):
    metrics = {"avg_dx_px": dx, "avg_dy_px": -0.5, "mean_kp_dist_before": 1.0, "mean_kp_dist_after": 0.5,
               **_band_metrics(improved),
               **tps_warp.describe_affine(np.array([[1.0, 0.0, dx], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]]), 100, 50)}
    result = SimpleNamespace(status=status, message="", metrics=metrics if status == "ok" else {},
                             diagnostics={"coarse_ransac_inliers": 40, "guided_ransac_inliers": guided})
    return registration.SceneRegistration(result=result, strategy="local", band_names=["blue", "green", "red", "nir"],
                                          band_map=[0, 1, 2, 3], target_mask_used=True, target_clear_fraction=0.9,
                                          reference_keypoint_scale=0.7, output_path=output_path)


def _stand_in(**per_model):
    """A register_scene_models stand-in that writes a small file to each
    requested output path (as the real one does for a successful
    registration) and returns _scene(**per_model[model])."""
    def register(target, reference, outputs, params, **kw):
        scenes = {}
        for model, path in outputs.items():
            with open(path, "wb") as f:
                f.write(model.encode())
            scenes[model] = _scene(output_path=path, **per_model.get(model, {}))
        return scenes
    return register


def test_register_one_returns_a_row_per_model_with_the_affine_transform(monkeypatch, tmp_path):
    monkeypatch.setattr(registration, "register_scene_models", _stand_in(affine={"dx": 0.25}, tps={"dx": 0.3}))
    task = _task(tmp_path)
    outcome = batch.register_one(task, registration.RegistrationParams())

    assert list(outcome["rows"]) == ["affine", "tps"]
    for model, row in outcome["rows"].items():
        assert row["warp_model"] == model and row["status"] == "ok"
        assert row["output_path"] == task["warp_outputs"][model] and row["output_written"]
        assert row["accepted"] == "true" and row["failed_checks"] == ""
        assert row["models_together"] == "affine+tps" and row["reference_keypoint_scale"] == 0.7
    affine = outcome["rows"]["affine"]
    assert (affine["affine_m00"], affine["affine_m02"], affine["affine_m12"]) == (1.0, 0.25, -0.5)
    assert affine["affine_shift_x_px"] == 0.25 and affine["affine_rotation_deg"] == 0.0
    assert outcome["rows"]["tps"]["avg_dx_px"] == 0.3
    assert set(tps_warp.AFFINE_METRIC_KEYS) <= set(batch.MANIFEST_FIELDS)


def test_register_one_records_an_exception_for_every_model(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise OSError("unreadable input")

    monkeypatch.setattr(registration, "register_scene_models", boom)
    outcome = batch.register_one(_task(tmp_path), registration.RegistrationParams())
    assert [row["status"] for row in outcome["rows"].values()] == ["error", "error"]
    assert all("unreadable input" in row["message"] for row in outcome["rows"].values())


def test_record_outcome_writes_each_model_to_its_own_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(registration, "register_scene_models", _stand_in())
    recorders = {m: batch.Recorder(str(tmp_path / m)) for m in ("affine", "tps")}
    batch.record_outcome(recorders, batch.register_one(_task(tmp_path), registration.RegistrationParams()))

    for model in ("affine", "tps"):
        with open(tmp_path / model / "registration_manifest.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and rows[0]["warp_model"] == model and rows[0]["status"] == "ok"
        assert float(rows[0]["affine_shift_x_px"]) == 0.25
        assert recorders[model].previously_recorded() == {"20240101_120000_abcd_3B_AnalyticMS_SR_harmonized_clip.tif": "ok"}
        metrics_csv = tmp_path / model / "alignment_metrics" / "registration_metrics_4band.csv"
        header, line = metrics_csv.read_text().splitlines()
        assert "affine_m00" in header.split(",") and len(header.split(",")) == len(line.split(","))
    # The scene's peak memory is logged once, with the first model.
    assert (tmp_path / "affine" / "worker_memory.csv").exists()
    assert not (tmp_path / "tps" / "worker_memory.csv").exists()


def test_with_staging_only_accepted_registrations_reach_the_output_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(registration, "register_scene_models",
                        _stand_in(affine={}, tps={"improved": False}))
    staging = tmp_path / "staging"
    task = {**_task(tmp_path), "staging_dir": str(staging),
            "acceptance": {"bands": ("red", "nir"), "max_abs_log_scale": 0.005, "min_ransac_inliers": 1000}}
    outcome = batch.register_one(task, registration.RegistrationParams())

    affine, tps = outcome["rows"]["affine"], outcome["rows"]["tps"]
    assert affine["accepted"] == "true" and affine["output_written"]
    assert affine["output_path"] == task["warp_outputs"]["affine"]
    assert open(task["warp_outputs"]["affine"], "rb").read() == b"affine"
    assert tps["accepted"] == "false" and not tps["output_written"] and tps["output_path"] == ""
    assert tps["failed_checks"] == "red_r2_not_improved;red_rmse_not_improved;nir_r2_not_improved;nir_rmse_not_improved"
    assert not os.path.exists(task["warp_outputs"]["tps"])
    assert os.listdir(staging) == []  # staged copies are always removed


def test_registration_failed_checks():
    names = ["blue", "green", "red", "nir"]
    good = {**_band_metrics(), "affine_scale_x": 1.004, "affine_scale_y": 0.996}
    assert registration.registration_failed_checks(good, names, 1001) == []
    assert registration.registration_failed_checks(good, names, 1000) == ["too_few_ransac_inliers"]
    assert registration.registration_failed_checks({**good, "affine_scale_y": 1.006}, names, 5000) == \
        ["scale_y_out_of_range"]
    assert registration.registration_failed_checks({**good, "affine_scale_x": 0.994}, names, 5000) == \
        ["scale_x_out_of_range"]
    # Only the whole-image ("_all") metrics count, and only for red and NIR.
    worse_inliers = {**good, "r2_band2_after_inliers": 0.1, "rmse_band0_after_all": 999.0}
    assert registration.registration_failed_checks(worse_inliers, names, 5000) == []
    assert registration.registration_failed_checks({**good, "rmse_band3_after_all": 300.0}, names, 5000) == \
        ["nir_rmse_not_improved"]
    # Strings from CSV are fine; a missing value fails its check.
    as_text = {k: str(v) for k, v in good.items()}
    assert registration.registration_failed_checks(as_text, names, "2000") == []
    missing = {k: v for k, v in good.items() if k != "r2_band2_after_all"}
    assert registration.registration_failed_checks(missing, names, 5000) == ["red_r2_not_improved"]
    # 8-band scenes: red and NIR are bands 5 and 7.
    eight = ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"]
    metrics8 = {"affine_scale_x": 1.0, "affine_scale_y": 1.0}
    for b in (5, 7):
        metrics8.update({f"r2_band{b}_before_all": 0.5, f"r2_band{b}_after_all": 0.6,
                         f"rmse_band{b}_before_all": 10, f"rmse_band{b}_after_all": 9})
    assert registration.registration_failed_checks(metrics8, eight, 5000) == []


def test_preferred_imagery_lists_accepted_scenes_as_registered(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "affine"
    recorder = batch.Recorder(str(out_dir))
    cases = {"20240101_120000_aaaa": {}, "20240101_120000_bbbb": {"improved": False},
             "20240101_120000_cccc": {"guided": 50}, "20240101_120000_dddd": {"dx": 0.1}}
    for scene_id, kwargs in cases.items():
        monkeypatch.setattr(registration, "register_scene_models", _stand_in(affine=kwargs))
        raw = raw_dir / f"{scene_id}_3B_AnalyticMS_SR_harmonized_clip.tif"
        raw.write_bytes(b"raw")
        task = {**_task(tmp_path, models=("affine",)), "scene_id": scene_id, "analytic_path": str(raw),
                "warp_outputs": {"affine": str(out_dir / "philadelphia4b" / "p" / raw.name)}}
        batch.record_outcome({"affine": recorder}, batch.register_one(task, registration.RegistrationParams()))
    # As in a run that saved every registration: the rejected ones' files exist too.
    path, rows = preferred_imagery.write_preferred_imagery(str(out_dir))

    by_id = {r["scene_id"]: r for r in rows}
    assert [by_id[s]["preferred_source"] for s in cases] == ["registered", "raw", "raw", "registered"]
    assert by_id["20240101_120000_bbbb"]["failed_checks"].startswith("red_r2_not_improved")
    assert by_id["20240101_120000_cccc"]["failed_checks"] == "too_few_ransac_inliers"
    assert by_id["20240101_120000_bbbb"]["preferred_analytic_path"] == by_id["20240101_120000_bbbb"]["raw_analytic_path"]
    assert by_id["20240101_120000_aaaa"]["preferred_analytic_path"].startswith(str(out_dir))
    assert float(by_id["20240101_120000_aaaa"]["r2_red_after_all"]) == pytest.approx(0.85)

    # The pipeline swaps only the analytic raster; UDM2/metadata stay raw.
    from psnorm import io
    scenes = [io.Scene(scene_id=s, analytic_path=str(raw_dir / f"{s}_3B_AnalyticMS_SR_harmonized_clip.tif"),
                       udm2_path=str(raw_dir / f"{s}_3B_udm2_clip.tif"), metadata_path=None, acquired=None)
              for s in list(cases) + ["20240101_120000_eeee"]]
    swapped = preferred_imagery.apply_preferred_imagery(scenes, path)
    assert [s.analytic_path.startswith(str(out_dir)) for s in swapped] == [True, False, False, True, False]
    assert [s.udm2_path for s in swapped] == [s.udm2_path for s in scenes]
    # Applying twice changes nothing more.
    assert preferred_imagery.apply_preferred_imagery(swapped, path) == swapped


def test_build_tasks_reuses_cached_scenes_and_leaves_out_ones_still_being_copied(monkeypatch, tmp_path):
    from datetime import datetime

    from psnorm import io

    folder = tmp_path / "philadelphia4b" / "philadelphia4b_2021_244_273"
    folder.mkdir(parents=True)
    paths = {}
    for name in ("old", "new", "copying"):
        paths[name] = str(folder / f"{name}.tif")
        open(paths[name], "w").close()
    # An hour on: "copying" was written just now (its mtime; the others' ctime is an hour old).
    later = os.path.getmtime(paths["copying"]) + 3600
    monkeypatch.setattr(batch.time, "time", lambda: later)
    os.utime(paths["copying"], (later, later))
    scenes = [io.Scene(scene_id=n, analytic_path=p, udm2_path=None, metadata_path=None,
                       acquired=datetime(2021, 9, 10)) for n, p in paths.items()]
    monkeypatch.setattr(batch.compositing, "discover_product_scenes",
                        lambda root, product, *a, **k: scenes if product == "philadelphia4b" else [])
    headers_read = []

    def raster_info(path):
        headers_read.append(path)
        return SimpleNamespace(width=10, height=20, band_count=4)

    monkeypatch.setattr(batch.io, "get_raster_info", raster_info)
    composites = [(2021, datetime(2021, 6, 21), "solstice_2021.tif")]
    cached = {paths["old"]: {"scene_id": "old", "analytic_path": paths["old"], "acquired": "2021-09-10T00:00:00"}}

    tasks = batch.build_tasks(str(tmp_path), str(tmp_path / "out"), composites, ["philadelphia4b", "philadelphia8b"],
                              cached=cached, min_file_age_s=600)

    assert [t["scene_id"] for t in tasks] == ["old", "new"]
    assert tasks[0] is cached[paths["old"]]
    assert headers_read == [paths["new"]]
    assert tasks[1]["reference_year"] == 2021 and tasks[1]["megapixels"] == 200 / 1e6


def test_a_4band_and_an_8band_product_of_one_capture_are_kept_apart(tmp_path):
    out_dir = tmp_path / "affine"
    recorder = batch.Recorder(str(out_dir))
    scene_id = "20200502_150055_00_2271"
    raws = {4: f"/raw/philadelphia4b/p/{scene_id}_3B_AnalyticMS_SR_harmonized_clip.tif",
            8: f"/raw/philadelphia8b/p/{scene_id}_3B_AnalyticMS_SR_8b_harmonized_clip.tif"}
    with open(recorder.manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=batch.MANIFEST_FIELDS)
        writer.writeheader()
        for n, raw in raws.items():
            writer.writerow({"scene_id": scene_id, "product": f"philadelphia{n}b", "analytic_path": raw,
                             "status": "insufficient_pixels", "band_count": n})

    assert recorder.previously_recorded() == {os.path.basename(raws[4]): "insufficient_pixels",
                                              os.path.basename(raws[8]): "insufficient_pixels"}
    rows = preferred_imagery.build_preferred_imagery(str(out_dir))
    assert sorted(r["raw_analytic_path"] for r in rows) == sorted(raws.values())
    assert {r["scene_id"] for r in rows} == {scene_id}
