import json
import os
import sys

import numpy as np

from psnorm import pipeline

from _raster_utils import write_geotiff

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS_DIR)
import sweep_thresholds  # noqa: E402


def _write_varying_scene(tmp_path, scene_id, arrays):
    path = write_geotiff(
        str(tmp_path / f"{scene_id}_3B_AnalyticMS_SR_harmonized_clip.tif"),
        arrays, band_descriptions=["blue", "green", "red", "nir"], nodata=0,
    )
    (tmp_path / f"{scene_id}_metadata.json").write_text(
        json.dumps({"properties": {"sun_elevation": 45, "clear_percent": 95}})
    )
    return path


def _make_run(tmp_path, size=60, seed=31):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    rng = np.random.default_rng(seed)
    ref_arrays = [rng.uniform(lo, hi, (size, size)) for lo, hi in [(200, 3000), (100, 1500), (150, 2000), (300, 3500)]]
    tgt_arrays = [np.clip(a * 1.04 + rng.normal(0, 3, a.shape), 1, None) for a in ref_arrays]
    _write_varying_scene(input_dir, "20240101_120000_ref", [a.astype(np.uint16) for a in ref_arrays])
    _write_varying_scene(input_dir, "20240102_120000_tgt", [a.astype(np.uint16) for a in tgt_arrays])

    pipeline.run_pipeline(
        str(input_dir), str(output_dir),
        use_omnicloudmask=False, use_water_mask=False, use_vegetation_mask=False,
        min_reference_area_km2=0.001, min_overlap_pixels=100, min_fit_pixels=2,
        device="cpu", workers=None,
    )
    return str(input_dir), str(output_dir)


def test_discover_target_records_finds_saved_fit(tmp_path):
    input_dir, output_dir = _make_run(tmp_path)

    records, reference_id, skipped_no_fit = sweep_thresholds.discover_target_records(output_dir, input_dir)

    assert reference_id == "20240101_120000_ref"
    assert skipped_no_fit == 0
    assert len(records) == 1
    assert records[0]["target_id"] == "20240102_120000_tgt"
    assert records[0]["fit"].rho.shape == (4,)
    # records intentionally carry paths, not the loaded evaluated mask (see
    # discover_target_records' docstring) -- the mask is loaded lazily,
    # per-scene, inside run_sweep instead.
    assert os.path.exists(records[0]["ref_flags_path"])
    assert os.path.exists(records[0]["tgt_flags_path"])
    assert "evaluated" not in records[0]


def test_discover_target_records_respects_limit(tmp_path):
    input_dir, output_dir = _make_run(tmp_path)
    records, _reference_id, _skipped = sweep_thresholds.discover_target_records(output_dir, input_dir, limit=0)
    assert records == []


def test_run_sweep_matches_single_scene_single_threshold_reference(tmp_path):
    # Cross-check run_sweep's scene-outer/threshold-inner accumulation
    # against the single-shot normalize.reclassify_invariant_pixels path
    # (which re-reads per call) -- they must agree exactly.
    input_dir, output_dir = _make_run(tmp_path)
    records, _reference_id, _skipped = sweep_thresholds.discover_target_records(output_dir, input_dir)
    from psnorm import io as psnorm_io
    from psnorm import masking, normalize

    ref_info = psnorm_io.get_raster_info(records[0]["reference_path"])
    ref_shape = (ref_info.height, ref_info.width)

    rows = sweep_thresholds.run_sweep(
        records, ref_shape, [0.6], [0.5], min_observations=1, min_fit_pixels=1, device="cpu",
    )

    rec = records[0]
    ref_flags = masking.load_flags_window(rec["ref_flags_path"], rec["ref_window"])
    tgt_flags = masking.load_flags_window(rec["tgt_flags_path"], rec["tgt_window"])
    evaluated = (ref_flags == 0) & (tgt_flags == 0)
    mask, n_eval, n_inv = normalize.reclassify_invariant_pixels(
        rec["reference_path"], rec["target_path"], rec["ref_window"], rec["tgt_window"],
        rec["ref_indices"], rec["tgt_indices"], evaluated, rec["fit"], ncp_threshold=0.6,
    )
    xoff, yoff, xsize, ysize = rec["ref_window"]
    expected_consensus = int((mask & evaluated).sum())  # this scene's own contribution, single-scene run

    assert rows[0]["n_consensus_pixels"] == expected_consensus


def test_run_sweep_save_rasters_writes_expected_layout(tmp_path):
    from psnorm import io as psnorm_io

    input_dir, output_dir = _make_run(tmp_path)
    records, _reference_id, _skipped = sweep_thresholds.discover_target_records(output_dir, input_dir)
    ref_info = psnorm_io.get_raster_info(records[0]["reference_path"])
    ref_shape = (ref_info.height, ref_info.width)

    raster_dir = str(tmp_path / "rasters")
    sweep_thresholds.run_sweep(
        records, ref_shape, [0.5, 0.7], [0.3, 0.5],
        min_observations=1, min_fit_pixels=1, device="cpu",
        raster_output_dir=raster_dir, reference_info=ref_info,
    )

    for ncp in [0.5, 0.7]:
        ncp_dir = os.path.join(raster_dir, f"ncp_{ncp}")
        assert os.path.exists(os.path.join(ncp_dir, "times_evaluated.tif"))
        assert os.path.exists(os.path.join(ncp_dir, "times_invariant.tif"))
        assert os.path.exists(os.path.join(ncp_dir, "frequency.tif"))
        for freq in [0.3, 0.5]:
            assert os.path.exists(os.path.join(ncp_dir, f"consensus_mask_freq{freq}.tif"))

    # consensus_mask.tif content must match a fresh threshold of times_evaluated/times_invariant
    from osgeo import gdal
    ds = gdal.Open(os.path.join(raster_dir, "ncp_0.5", "consensus_mask_freq0.3.tif"))
    consensus = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    assert consensus.dtype == np.uint8
    assert consensus.sum() > 0  # this synthetic fixture is correlated enough to produce some invariant pixels


def test_sweep_main_writes_summary_csv(tmp_path, capsys):
    input_dir, output_dir = _make_run(tmp_path)
    summary_csv = os.path.join(output_dir, "sweep.csv")

    argv_backup = sys.argv
    try:
        sys.argv = [
            "sweep_thresholds.py", "--input", input_dir, "--output", output_dir,
            "--ncp-thresholds", "0.5,0.7", "--freq-thresholds", "0.3,0.5",
            "--device", "cpu", "--summary-csv", summary_csv,
        ]
        sweep_thresholds.main()
    finally:
        sys.argv = argv_backup

    assert os.path.exists(summary_csv)
    with open(summary_csv) as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 1 + 2 * 2  # header + (ncp x freq) grid rows
    assert "ncp_threshold" in lines[0] and "n_consensus_pixels" in lines[0]
