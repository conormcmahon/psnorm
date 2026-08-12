#!/usr/bin/env python3
"""Run the psnorm pipeline against the PlanetScope test imagery and print the
adjacent-pair R^2/RMSE before/after summary.

Usage:
    .venv/bin/python scripts/run_test_images.py [--no-omnicloudmask] [--workers N]
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
