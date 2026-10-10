"""Image co-registration between a target scene and the reference scene.

IR-MAD (irmad.py) requires the reference and target to be readable on a
shared pixel grid -- `pipeline.py` gets this for free when the two rasters
are already grid-aligned (see `io.grids_aligned`), which holds for
same-collection PlanetScope strips sharing a common ortho tiling grid.
`register_to_reference` is the real warp/align step, built on top of the
sensor/CRS-agnostic `planet_georeg_opencv` library (SIFT keypoint matching
over normalized-difference index channels, RANSAC, and either a local TPS
warp or a global affine warp -- see that library's own docs for how the
matching/warping actually works). This module only adds the PlanetScope-
and psnorm-specific pieces on top: band-name resolution (via
`psnorm.sensors`), composite-reference support (via
`psnorm.composite_reference`, for registering a tile against a multi-
statistic composite instead of a single reference scene), and picking
which of the library's two warp strategies keeps the result grid-aligned
with the reference.

`is_grid_aligned` only checks CRS/pixel-size/pixel-integer-origin-offset --
it does NOT detect genuine sub-pixel/few-pixel geolocation error, which is
PlanetScope's well-known weak point even between same-collection strips.
Callers that want registration to actually correct that error (as opposed
to only reprojecting mismatched grids onto each other) should call
`register_to_reference` unconditionally rather than gating it on
`is_grid_aligned` -- see pipeline.py's `force_registration` option.
"""

from __future__ import annotations

import os

import planet_georeg_opencv as pgo

from . import composite_reference, io, sensors


class RegistrationFailed(Exception):
    """Raised when register_to_reference cannot produce an aligned raster
    (insufficient overlap, too few keypoint matches, RANSAC failure, etc),
    as opposed to a configuration error (e.g. a bad path)."""


def is_grid_aligned(reference_info: io.RasterInfo, target_info: io.RasterInfo, tol: float = 1e-6) -> bool:
    """True if `target_info` can be directly overlapped with
    `reference_info` (see `io.overlap_window`) with no resampling."""
    return io.grids_aligned(reference_info, target_info, tol=tol)


def register_to_reference(
    target_path: str,
    reference_path: str,
    *,
    output_path: str | None = None,
    reference_stat: str = "median",
    warp_model: str = "tps",
    buffer_m: float = 500.0,
) -> str:
    """Co-register `target_path` onto `reference_path`.

    `reference_path` may be a plain analytic scene (resolved via
    `sensors.detect_band_names`) or a compositing.py-produced multi-
    statistic composite (detected via `composite_reference.is_composite`
    and resolved by extracting just the `{band}_{reference_stat}` bands,
    windowed to `target_path`'s own extent -- see composite_reference.py
    for why this never reads more than a small window of a composite that
    may be tens of GB on disk).

    Band correspondence is resolved by logical band name (blue/green/red/
    nir/...), not by position -- this supersedes the original stub's
    `band_index` parameter, which assumed a single-band registration
    method; the actual method needs at least two corresponding bands to
    compute a normalized-difference index.

    Two warp strategies are available in planet_georeg_opencv, and the
    right one depends on whether `target_path` and `reference_path`
    already share a grid:
      - If they do (the common case for scenes/composites built from the
        same raw archive and ortho tiling grid): `pgo.register_local`, a
        self-referential local (TPS) warp that corrects `target_path`
        onto ITS OWN pixel grid -- the output's geotransform is literally
        unchanged, so it trivially stays grid-aligned with the reference.
      - If they don't (different CRS/pixel size/origin phase):
        `pgo.register_to_grid`, a global affine warp onto the reference's
        own full pixel grid, so the result satisfies `is_grid_aligned`
        against it afterward as this function's contract requires.

    `warp_model` ("tps" or "affine") picks the correction used in the
    grid-aligned case (see pgo.register_local): "tps" corrects spatially
    varying error but can bend areas far from any matched keypoint;
    "affine" applies one global shift/rotation/scale/shear and is more
    stable with sparse or clustered matches. It has no effect in the
    not-grid-aligned case, which is always a global affine.

    `buffer_m` limits the reference area searched for keypoints to
    `target_path`'s own extent plus this distance on every side (metres
    for a projected CRS). For a composite reference it also bounds how
    much of the composite is read at all. It has no effect in the
    not-grid-aligned case.

    Returns the path to the registered raster (`output_path`, or a
    derived default next to `target_path` if not given). Preserves band
    count, band order, and nodata handling from `target_path`.

    Raises:
        RegistrationFailed: if registration could not be completed (e.g.
            no geographic overlap, too few keypoint matches, or fewer
            than two shared bands between target and reference).
    """
    if warp_model not in ("tps", "affine"):
        raise ValueError(f"warp_model must be 'tps' or 'affine', got {warp_model!r}")
    target_info = io.get_raster_info(target_path)
    target_names = sensors.detect_band_names(target_path)

    if composite_reference.is_composite(reference_path):
        reference_input = composite_reference.extract_reference_window(
            reference_path, target_info, target_names, stat=reference_stat, buffer_m=buffer_m,
        )
        reference_names = list(target_names)
    else:
        reference_input = reference_path
        reference_names = sensors.detect_band_names(reference_path)
    reference_info = io.get_raster_info(reference_path)

    band_map = [reference_names.index(name) if name in reference_names else -1 for name in target_names]
    if sum(1 for b in band_map if b != -1) < 2:
        raise RegistrationFailed(
            f"Fewer than two shared bands between target {target_names!r} "
            f"and reference {reference_names!r}; cannot compute a "
            f"normalized-difference index to register on."
        )

    if is_grid_aligned(reference_info, target_info):
        # fixed_buffer_m keeps the library from trimming the reference back
        # down to the strict overlap before keypoint detection.
        result = pgo.register_local(
            target_path, reference_input, band_map, warp_model=warp_model, fixed_buffer_m=buffer_m,
        )
    else:
        result = pgo.register_to_grid(target_path, reference_input, band_map)

    if result.status != "ok":
        raise RegistrationFailed(f"register_to_reference('{target_path}'): {result.status} -- {result.message}")

    if output_path is None:
        base, ext = os.path.splitext(target_path)
        output_path = f"{base}_registered{ext}"
    result.registered.rio.to_raster(output_path)
    return output_path
