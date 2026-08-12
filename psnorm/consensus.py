"""Cross-scene consensus aggregation of invariant-target candidates.

Every candidate comes from a target-vs-reference IR-MAD comparison (see
normalize.detect_invariant_candidates), so every candidate pixel is by
construction inside the reference scene's own footprint (`ref_window` is
always a sub-window of the reference). The reference's own raster grid is
therefore used directly as the shared coordinate frame — no separate
union-grid computation is needed, unlike a design that compared every
overlapping pair of images to each other.

For each reference pixel, two counts are accumulated across all target
comparisons:
  times_evaluated: number of target comparisons where this pixel passed the
                   search mask (nodata/water/cloud excluded) in *both* the
                   reference and that target scene
  times_invariant: subset of times_evaluated where IR-MAD flagged the pixel
                   as an invariant candidate for that comparison

"evaluated" is recomputed here from each scene's saved exclusion-flags
raster (masking.compute_exclusion_flags), not stored per-pair — so a pixel
masked out in a given comparison contributes to neither counter for that
comparison ("don't count masked pixels against a pixel's percentage of
invariance").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from osgeo import gdal

from . import io, masking

gdal.UseExceptions()


@dataclass
class ConsensusRecord:
    """One target's Phase A result, referenced by path rather than held in
    memory — consensus aggregation reads each pair's arrays back from disk
    one at a time so peak memory stays bounded by one pair, not all pairs."""

    target_id: str
    ref_window: io.Window
    tgt_window: io.Window
    candidate_mask_path: str
    reference_flags_path: str
    target_flags_path: str


@dataclass
class ConsensusResult:
    times_evaluated: np.ndarray
    times_invariant: np.ndarray
    frequency: np.ndarray  # float32, NaN where times_evaluated == 0
    consensus_mask: np.ndarray  # boolean


def aggregate_consensus(
    reference_shape: tuple[int, int],
    records: list[ConsensusRecord],
    *,
    invariance_frequency_threshold: float,
    min_observations: int,
) -> ConsensusResult:
    """Accumulate times_evaluated/times_invariant over `records` into arrays
    shaped like the reference scene, then threshold into a final consensus
    mask: `frequency > invariance_frequency_threshold` AND at least
    `min_observations` comparisons actually evaluated that pixel (a floor
    so a pixel seen once or twice can't reach 100% by chance).
    """
    height, width = reference_shape
    times_evaluated = np.zeros((height, width), dtype=np.uint16)
    times_invariant = np.zeros((height, width), dtype=np.uint16)

    for record in records:
        xoff, yoff, xsize, ysize = record.ref_window
        ref_flags = masking.load_flags_window(record.reference_flags_path, record.ref_window)
        tgt_flags = masking.load_flags_window(record.target_flags_path, record.tgt_window)
        evaluated = (ref_flags == 0) & (tgt_flags == 0)

        candidate_dataset, candidate_bands = io.open_bands(record.candidate_mask_path, band_indices=[1])
        candidate = candidate_bands[0].ReadAsArray(0, 0, xsize, ysize) != 0
        candidate_dataset = None

        times_evaluated[yoff : yoff + ysize, xoff : xoff + xsize] += evaluated.astype(np.uint16)
        times_invariant[yoff : yoff + ysize, xoff : xoff + xsize] += (evaluated & candidate).astype(np.uint16)

    with np.errstate(invalid="ignore", divide="ignore"):
        frequency = np.where(
            times_evaluated > 0,
            times_invariant.astype(np.float32) / np.maximum(times_evaluated, 1),
            np.nan,
        ).astype(np.float32)

    consensus_mask = (times_evaluated >= min_observations) & (frequency > invariance_frequency_threshold)

    return ConsensusResult(
        times_evaluated=times_evaluated,
        times_invariant=times_invariant,
        frequency=frequency,
        consensus_mask=consensus_mask,
    )


def save_consensus(result: ConsensusResult, reference_info: io.RasterInfo, output_dir: str) -> dict:
    """Persist all four consensus arrays as GeoTIFFs on the reference's
    grid. Returns a dict of the written paths."""
    import os

    paths = {
        "times_evaluated": os.path.join(output_dir, "times_evaluated.tif"),
        "times_invariant": os.path.join(output_dir, "times_invariant.tif"),
        "frequency": os.path.join(output_dir, "frequency.tif"),
        "consensus_mask": os.path.join(output_dir, "consensus_mask.tif"),
    }
    io.write_single_band_raster(result.times_evaluated, reference_info, paths["times_evaluated"], gdal.GDT_UInt16)
    io.write_single_band_raster(result.times_invariant, reference_info, paths["times_invariant"], gdal.GDT_UInt16)
    io.write_single_band_raster(result.frequency, reference_info, paths["frequency"], gdal.GDT_Float32)
    io.write_single_band_raster(result.consensus_mask.astype(np.uint8), reference_info, paths["consensus_mask"], gdal.GDT_Byte)
    return paths
