import numpy as np
import pytest
from osgeo import gdal

from psnorm import consensus, io, masking

from _raster_utils import write_geotiff


def _byte_raster(path, value):
    return write_geotiff(path, [np.array([[value]], dtype=np.uint8)], dtype=gdal.GDT_Byte, nodata=None)


def test_masked_pixels_excluded_from_frequency_denominator(tmp_path):
    # Matches the spec example exactly: a pixel flagged invariant-candidate
    # 8 times, unmasked-but-not-invariant 2 times, and masked (cloud) 3 more
    # times -> percent invariant should be 8/(8+2) = 80%, not 8/13.
    ref_flags_path = _byte_raster(str(tmp_path / "ref_flags.tif"), 0)  # reference always clear here

    records = []
    for i in range(8):  # unmasked, invariant candidate
        tgt_flags = _byte_raster(str(tmp_path / f"tgt_flags_inv_{i}.tif"), 0)
        candidate = _byte_raster(str(tmp_path / f"cand_inv_{i}.tif"), 1)
        records.append(consensus.ConsensusRecord(
            target_id=f"inv_{i}", ref_window=(0, 0, 1, 1), tgt_window=(0, 0, 1, 1),
            candidate_mask_path=candidate, reference_search_mask_path=ref_flags_path, target_search_mask_path=tgt_flags,
        ))
    for i in range(2):  # unmasked, NOT invariant
        tgt_flags = _byte_raster(str(tmp_path / f"tgt_flags_notinv_{i}.tif"), 0)
        candidate = _byte_raster(str(tmp_path / f"cand_notinv_{i}.tif"), 0)
        records.append(consensus.ConsensusRecord(
            target_id=f"notinv_{i}", ref_window=(0, 0, 1, 1), tgt_window=(0, 0, 1, 1),
            candidate_mask_path=candidate, reference_search_mask_path=ref_flags_path, target_search_mask_path=tgt_flags,
        ))
    for i in range(3):  # masked (cloud) -- candidate=1 on purpose, must still not count
        tgt_flags = _byte_raster(str(tmp_path / f"tgt_flags_masked_{i}.tif"), masking.EXCLUDE_UDM2)
        candidate = _byte_raster(str(tmp_path / f"cand_masked_{i}.tif"), 1)
        records.append(consensus.ConsensusRecord(
            target_id=f"masked_{i}", ref_window=(0, 0, 1, 1), tgt_window=(0, 0, 1, 1),
            candidate_mask_path=candidate, reference_search_mask_path=ref_flags_path, target_search_mask_path=tgt_flags,
        ))

    result = consensus.aggregate_consensus(
        (1, 1), records, invariance_frequency_threshold=0.7, min_observations=2,
    )

    assert result.times_evaluated[0, 0] == 10  # 8 + 2, the 3 masked ones excluded
    assert result.times_invariant[0, 0] == 8
    assert result.frequency[0, 0] == pytest.approx(0.8)
    assert result.consensus_mask[0, 0] == True  # 0.8 > 0.7 threshold


def test_frequency_threshold_is_respected(tmp_path):
    ref_flags_path = _byte_raster(str(tmp_path / "ref_flags.tif"), 0)
    records = []
    for i in range(8):
        tgt_flags = _byte_raster(str(tmp_path / f"tgt_flags_a_{i}.tif"), 0)
        candidate = _byte_raster(str(tmp_path / f"cand_a_{i}.tif"), 1)
        records.append(consensus.ConsensusRecord(
            target_id=f"a_{i}", ref_window=(0, 0, 1, 1), tgt_window=(0, 0, 1, 1),
            candidate_mask_path=candidate, reference_search_mask_path=ref_flags_path, target_search_mask_path=tgt_flags,
        ))
    for i in range(2):
        tgt_flags = _byte_raster(str(tmp_path / f"tgt_flags_b_{i}.tif"), 0)
        candidate = _byte_raster(str(tmp_path / f"cand_b_{i}.tif"), 0)
        records.append(consensus.ConsensusRecord(
            target_id=f"b_{i}", ref_window=(0, 0, 1, 1), tgt_window=(0, 0, 1, 1),
            candidate_mask_path=candidate, reference_search_mask_path=ref_flags_path, target_search_mask_path=tgt_flags,
        ))

    # frequency is 0.8 here -> passes a 0.7 threshold, fails a 0.85 threshold
    lenient = consensus.aggregate_consensus((1, 1), records, invariance_frequency_threshold=0.7, min_observations=2)
    strict = consensus.aggregate_consensus((1, 1), records, invariance_frequency_threshold=0.85, min_observations=2)
    assert lenient.consensus_mask[0, 0] == True
    assert strict.consensus_mask[0, 0] == False


def test_min_observations_floor_rejects_low_sample_pixels(tmp_path):
    ref_flags_path = _byte_raster(str(tmp_path / "ref_flags.tif"), 0)
    tgt_flags = _byte_raster(str(tmp_path / "tgt_flags.tif"), 0)
    candidate = _byte_raster(str(tmp_path / "cand.tif"), 1)
    # A single 100%-invariant observation should NOT pass a min_observations
    # floor above 1, even though frequency == 1.0.
    records = [consensus.ConsensusRecord(
        target_id="only_one", ref_window=(0, 0, 1, 1), tgt_window=(0, 0, 1, 1),
        candidate_mask_path=candidate, reference_search_mask_path=ref_flags_path, target_search_mask_path=tgt_flags,
    )]
    result = consensus.aggregate_consensus((1, 1), records, invariance_frequency_threshold=0.7, min_observations=3)
    assert result.frequency[0, 0] == 1.0
    assert result.consensus_mask[0, 0] == False
