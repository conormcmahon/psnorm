#!/usr/bin/env python3
"""Write preferred_imagery.csv for an existing registration output directory
(one written by scripts/register_to_solstice_composites.py, which also
writes this file itself at the end of every run): for every scene, the
registered version if it passes psnorm.registration.RegistrationAcceptance,
otherwise the raw scene. See psnorm.preferred_imagery.

Nothing is moved or deleted; registered scenes that fail the checks stay on
disk, they are just not listed as preferred.

Usage:
    python scripts/write_preferred_imagery.py \\
        --output-dir /mnt/e/E_Coast_PlanetScope/philadelphia/affine_transform
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import preferred_imagery, registration  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="a registration output directory (holds registration_manifest.csv)")
    parser.add_argument("--manifest", default=None,
                        help=f"where to write it (default: {preferred_imagery.PREFERRED_IMAGERY_FILENAME} in --output-dir)")
    registration.add_acceptance_arguments(parser)
    args = parser.parse_args()

    path, rows = preferred_imagery.write_preferred_imagery(
        args.output_dir, registration.acceptance_from_args(args), path=args.manifest)
    sources = Counter(r["preferred_source"] for r in rows)
    print(f"{len(rows)} scenes: {sources['registered']} registered, {sources['raw']} raw. Wrote {path}")
    reasons = Counter(check for r in rows for check in r["failed_checks"].split(";") if check)
    for check, n in reasons.most_common():
        print(f"  {check}: {n}")


if __name__ == "__main__":
    main()
