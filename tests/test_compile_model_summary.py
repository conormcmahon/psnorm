import json
import os
import sys

import numpy as np

from psnorm import pipeline

from _raster_utils import write_geotiff

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS_DIR)
import compile_model_summary  # noqa: E402


def _write_varying_scene(tmp_path, scene_id, arrays):
    path = write_geotiff(
        str(tmp_path / f"{scene_id}_3B_AnalyticMS_SR_harmonized_clip.tif"),
        arrays, band_descriptions=["blue", "green", "red", "nir"], nodata=0,
    )
    (tmp_path / f"{scene_id}_metadata.json").write_text(
        json.dumps({"properties": {"sun_elevation": 45, "clear_percent": 95}})
    )
    return path


def _make_run(tmp_path, size=60, seed=41):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    rng = np.random.default_rng(seed)
    ref_arrays = [rng.uniform(lo, hi, (size, size)) for lo, hi in [(200, 3000), (100, 1500), (150, 2000), (300, 3500)]]
    tgt_arrays = [np.clip(a * 1.04 + rng.normal(0, 3, a.shape), 1, None) for a in ref_arrays]
    _write_varying_scene(input_dir, "20240101_120000_ref", [a.astype(np.uint16) for a in ref_arrays])
    _write_varying_scene(input_dir, "20240215_083000_tgt", [a.astype(np.uint16) for a in tgt_arrays])

    pipeline.run_pipeline(
        str(input_dir), str(output_dir),
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        invariance_frequency_threshold=0.1, min_observations=1,
        device="cpu", workers=None,
    )
    return str(input_dir), str(output_dir)


def test_compile_rows_one_row_per_band_per_scene(tmp_path):
    input_dir, output_dir = _make_run(tmp_path)

    rows, n_with_model, n_without_model = compile_model_summary.compile_rows(output_dir, input_dir)

    # reference (identity) + target scene, 4 bands each
    assert n_with_model == 2
    assert n_without_model == 0
    assert len(rows) == 8

    target_rows = [r for r in rows if r["target_id"] == "20240215_083000_tgt"]
    assert len(target_rows) == 4
    assert {r["band_name"] for r in target_rows} == {"blue", "green", "red", "nir"}
    assert all(r["date"] == "2024-02-15" for r in target_rows)
    assert all(r["datetime"].startswith("2024-02-15T08:30:00") for r in target_rows)
    assert all(r["reference_id"] == "20240101_120000_ref" for r in target_rows)

    ref_rows = [r for r in rows if r["target_id"] == "20240101_120000_ref"]
    assert len(ref_rows) == 4
    assert all(r["date"] == "2024-01-01" for r in ref_rows)
    assert all(float(r["slope"]) == 1.0 and float(r["intercept"]) == 0.0 for r in ref_rows)


def test_compile_rows_reports_scenes_missing_a_model(tmp_path):
    input_dir, output_dir = _make_run(tmp_path)
    # a third scene discoverable in --input but never run through the
    # pipeline (no model.json on disk) -- e.g. added after the run.
    rng = np.random.default_rng(99)
    extra_arrays = [rng.uniform(lo, hi, (60, 60)).astype(np.uint16) for lo, hi in [(200, 3000), (100, 1500), (150, 2000), (300, 3500)]]
    _write_varying_scene(tmp_path / "input", "20240301_120000_extra", extra_arrays)

    rows, n_with_model, n_without_model = compile_model_summary.compile_rows(output_dir, input_dir)
    assert n_with_model == 2
    assert n_without_model == 1
    assert "20240301_120000_extra" not in {r["target_id"] for r in rows}


def test_main_writes_csv(tmp_path):
    input_dir, output_dir = _make_run(tmp_path)
    csv_path = os.path.join(output_dir, "summary.csv")

    argv_backup = sys.argv
    try:
        sys.argv = ["compile_model_summary.py", "--input", input_dir, "--output", output_dir, "--csv", csv_path]
        compile_model_summary.main()
    finally:
        sys.argv = argv_backup

    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 1 + 8  # header + 8 band-rows
    assert lines[0] == ",".join(compile_model_summary.FIELDNAMES)
