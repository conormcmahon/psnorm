#!/usr/bin/env python3
"""Run the psnorm pipeline against the PlanetScope test imagery and print the
adjacent-pair R^2/RMSE before/after summary.

Usage:
    .venv/bin/python scripts/run_test_images.py [--no-omnicloudmask] [--workers N] [--device auto|cpu|gpu] \\
        [--ncp-threshold F] [--invariance-frequency-threshold F] \\
        [--downsample-targets] [--downsample-resolution-m F]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm.pipeline import run_pipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INPUT = os.path.join(REPO_ROOT, "test_images", "output_20241001_4b", "files")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "test_images", "output_20241001_4b", "psnorm_output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-omnicloudmask", action="store_true")
    parser.add_argument("--workers", default="cpu")
    parser.add_argument("--resume", default="validate", choices=["no", "yes", "validate"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--ncp-threshold", type=float, default=0.70)
    parser.add_argument("--invariance-frequency-threshold", type=float, default=0.5)
    parser.add_argument("--reference-path", default=None, help="force this scene as reference (skips auto-selection)")
    parser.add_argument("--downsample-targets", action="store_true",
                         help="search for targets / fit correction factors at reduced resolution (see --downsample-resolution-m)")
    parser.add_argument("--downsample-resolution-m", type=float, default=15.0)
    parser.add_argument("--dsm-path", default=None,
                         help="optional digital surface model (single file or a directory of tiles) enabling the "
                              "horizontality/roughness/shadow LiDAR masks; omit to disable them entirely")
    parser.add_argument("--dsm-height-units", default="m", choices=["m", "ft"])
    parser.add_argument("--max-slope-deg", type=float, default=5.0)
    parser.add_argument("--roughness-window-radius-px", type=int, default=2)
    parser.add_argument("--roughness-max-deg", type=float, default=10.0)
    parser.add_argument("--use-shadow-mask", action="store_true",
                         help="also ray-trace each scene's own sun position against the DSM (expensive; opt-in "
                              "separately from --dsm-path)")
    parser.add_argument("--max-building-height-m", type=float, default=150.0)
    parser.add_argument("--shadow-ray-step-m", type=float, default=None,
                         help="shadow ray-march step size; defaults to one native DSM pixel")
    parser.add_argument("--shadow-downsample-factor", type=int, default=1,
                         help="NOT YET SUPPORTED (ignored with a warning; always runs at native resolution) -- "
                              "was meant as a working-resolution factor for the shadow ray-trace specifically "
                              "(independent of --downsample-targets), but the row-blocked implementation that "
                              "keeps shadow masking memory-safe doesn't support it yet")
    parser.add_argument("--shadow-angle-bucket-deg", type=float, default=1.0,
                         help="sun (azimuth, elevation) rounding granularity for the on-disk shadow-mask cache")
    parser.add_argument("--mask-erode-px", type=int, default=1)
    parser.add_argument("--mask-dilate-px", type=int, default=1)
    args = parser.parse_args()

    workers = args.workers
    if workers not in ("cpu", "none", None):
        workers = int(workers)
    elif workers == "none":
        workers = None

    start = time.time()
    result = run_pipeline(
        args.input,
        args.output,
        use_omnicloudmask=not args.no_omnicloudmask,
        workers=workers,
        resume=args.resume,
        device=args.device,
        ncp_threshold=args.ncp_threshold,
        invariance_frequency_threshold=args.invariance_frequency_threshold,
        reference_path=args.reference_path,
        downsample_targets=args.downsample_targets,
        downsample_resolution_m=args.downsample_resolution_m,
        dsm_path=args.dsm_path,
        dsm_height_units=args.dsm_height_units,
        max_slope_deg=args.max_slope_deg,
        roughness_window_radius_px=args.roughness_window_radius_px,
        roughness_max_deg=args.roughness_max_deg,
        use_shadow_mask=args.use_shadow_mask,
        max_building_height_m=args.max_building_height_m,
        shadow_ray_step_m=args.shadow_ray_step_m,
        shadow_downsample_factor=args.shadow_downsample_factor,
        shadow_angle_bucket_deg=args.shadow_angle_bucket_deg,
        mask_erode_px=args.mask_erode_px,
        mask_dilate_px=args.mask_dilate_px,
    )
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s. Reference: {result.reference_scene_id}")
    print(f"Report written to: {os.path.join(args.output, 'summary.md')}")
    print(f"CSV written to:    {os.path.join(args.output, 'adjacent_pair_report.csv')}")
    print()
    with open(os.path.join(args.output, "summary.md")) as f:
        print(f.read())


if __name__ == "__main__":
    main()
