"""Image co-registration between a target scene and the reference scene.

IR-MAD (irmad.py) requires the reference and target to be readable on a
shared pixel grid — `pipeline.py` gets this for free when the two rasters
are already grid-aligned (see `io.grids_aligned`), which holds for
same-collection PlanetScope strips sharing a common ortho tiling grid. When
it doesn't hold (different sensors, different processing runs, or genuine
sub-pixel misregistration), `register_to_reference` is the place a real
warp/align step belongs. It is intentionally left unimplemented for now.
"""

from __future__ import annotations

from . import io


def is_grid_aligned(reference_info: io.RasterInfo, target_info: io.RasterInfo, tol: float = 1e-6) -> bool:
    """True if `target_info` can be directly overlapped with
    `reference_info` (see `io.overlap_window`) with no resampling."""
    return io.grids_aligned(reference_info, target_info, tol=tol)


def register_to_reference(
    target_path: str,
    reference_path: str,
    *,
    output_path: str | None = None,
    band_index: int = 1,
) -> str:
    """Co-register `target_path` onto `reference_path`'s pixel grid.

    STUB — not yet implemented.

    Intended contract, for whoever implements this next:
      - Input: paths to a target raster and a reference raster that are
        *not* grid-aligned (different CRS, pixel size, and/or a
        non-pixel-integer origin offset — see `io.grids_aligned`).
      - Output: path to a new raster (at `output_path`, or a derived default
        next to `target_path` if not given) resampled/warped onto the
        reference's CRS, pixel size, and origin. Must preserve band count,
        band order, and nodata value/handling from `target_path`.
      - `band_index` names which 1-based band to use for estimating the
        registration transform (e.g. a feature-matching or
        phase-correlation step, computed on a single band), independent of
        which/how-many bands get warped in the output.
      - Called by `pipeline.py` only when
        `is_grid_aligned(reference_info, target_info)` is False, and the
        result is expected to satisfy `is_grid_aligned` against the
        reference afterwards.

    Raises:
        NotImplementedError: always, until this is implemented.
    """
    raise NotImplementedError(
        "register_to_reference() is a placeholder — image co-registration "
        "is not implemented yet. This target scene is not grid-aligned "
        "with the reference and cannot be normalized until either this "
        "function is implemented or the input data is pre-aligned."
    )
