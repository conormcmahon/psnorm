#!/usr/bin/env python3
"""Compile every scene's model.json (see model_io.save_model) in a completed
psnorm run into one long-format CSV: one row per (target_id, band), with the
acquisition date parsed from target_id (see io.parse_acquisition_time) and
every BandModel/NormalizationModel field carried through.

Scans scenes via io.discover_scenes(--input) and checks each expected
models/{scene_id}_model.json path directly (not a directory listing over
models/) -- same reasoning as sweep_thresholds.py's discover_target_records:
on this environment's external-drive mount, directory listings have shown
staleness relative to what's actually readable by path, so deriving the
expected scene set from --input and checking existence per-path is more
reliable than glob/listdir over the output directory.

Includes the reference scene's own (identity) model -- it has a model.json
just like every fitted target, with slope=1.0/intercept=0.0/r2=1.0 per band
(see normalize.identity_model) -- so it shows up as one ordinary row per
band rather than needing special-casing.

Usage:
    .venv/bin/python scripts/compile_model_summary.py \\
        --input /path/to/scenes --output /path/to/completed_psnorm_output \\
        --csv model_summary_long.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import io, model_io  # noqa: E402

FIELDNAMES = [
    "target_id", "date", "datetime", "reference_id", "band_name",
    "slope", "intercept", "r2", "n_invariant_pixels", "n_outliers_excluded",
    "rmse_before", "rmse_after", "identity_fallback", "fallback_reason",
    "raw_slope", "raw_intercept",
    "n_consensus_pixels", "invariance_frequency_threshold", "min_observations", "fitted_at",
]


def compile_rows(output_dir: str, input_dir: str, *, log=print) -> tuple[list[dict], int, int]:
    """Returns (rows, n_scenes_with_model, n_scenes_without_model)."""
    scenes = io.discover_scenes(input_dir)
    rows = []
    n_with_model = 0
    n_without_model = 0

    for scene in scenes:
        model_path = os.path.join(output_dir, "models", f"{scene.scene_id}_model.json")
        if not os.path.exists(model_path):
            n_without_model += 1  # not fitted -- error/skipped_*/not yet run
            continue
        try:
            model = model_io.load_model(model_path)
        except Exception as exc:
            log(f"  skipping unparseable {model_path}: {exc}")
            n_without_model += 1
            continue
        n_with_model += 1

        acquired = io.parse_acquisition_time(model.target_id)
        date_str = acquired.date().isoformat() if acquired else ""
        datetime_str = acquired.isoformat() if acquired else ""

        for band in model.bands:
            rows.append({
                "target_id": model.target_id,
                "date": date_str,
                "datetime": datetime_str,
                "reference_id": model.reference_id,
                "band_name": band.band_name,
                "slope": band.slope,
                "intercept": band.intercept,
                "r2": band.r2,
                "n_invariant_pixels": band.n_invariant_pixels,
                "n_outliers_excluded": band.n_outliers_excluded,
                "rmse_before": band.rmse_before,
                "rmse_after": band.rmse_after,
                "identity_fallback": band.identity_fallback,
                "fallback_reason": band.fallback_reason,
                "raw_slope": band.raw_slope,
                "raw_intercept": band.raw_intercept,
                "n_consensus_pixels": model.n_consensus_pixels,
                "invariance_frequency_threshold": model.invariance_frequency_threshold,
                "min_observations": model.min_observations,
                "fitted_at": model.fitted_at,
            })

    rows.sort(key=lambda r: (r["date"], r["target_id"], r["band_name"]))
    return rows, n_with_model, n_without_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="original scene folder (authoritative scene id list)")
    parser.add_argument("--output", required=True, help="completed psnorm_output dir (models/)")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(args.output, "model_summary_long.csv")

    print(f"Discovering scenes in {args.input}...")
    rows, n_with_model, n_without_model = compile_rows(args.output, args.input)
    print(f"  {n_with_model} scenes with a model.json, {n_without_model} without "
          f"(not fitted / error / skipped) -> {len(rows)} band-rows")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to: {csv_path}")


if __name__ == "__main__":
    main()
