import json
from datetime import datetime

import numpy as np
import pytest
from osgeo import gdal

from psnorm import io

from _raster_utils import write_geotiff


def _info(origin, size, pixel_size=3.0):
    arr = np.zeros(size, dtype=np.uint16)
    return arr, origin, pixel_size


def test_parse_acquisition_time():
    dt = io.parse_acquisition_time("20241005_160142_87_24d6")
    assert dt == datetime(2024, 10, 5, 16, 1, 42)


def test_parse_acquisition_time_invalid():
    assert io.parse_acquisition_time("not_a_timestamp") is None


def test_grids_aligned_true_for_pixel_integer_offset(tmp_path):
    arr = np.zeros((10, 10), dtype=np.uint16)
    path_a = write_geotiff(str(tmp_path / "a.tif"), [arr], origin=(0.0, 100.0), pixel_size=3.0)
    path_b = write_geotiff(str(tmp_path / "b.tif"), [arr], origin=(9.0, 91.0), pixel_size=3.0)  # offset by 3,3 px
    info_a, info_b = io.get_raster_info(path_a), io.get_raster_info(path_b)
    assert io.grids_aligned(info_a, info_b)


def test_grids_aligned_false_for_subpixel_offset(tmp_path):
    arr = np.zeros((10, 10), dtype=np.uint16)
    path_a = write_geotiff(str(tmp_path / "a.tif"), [arr], origin=(0.0, 100.0), pixel_size=3.0)
    path_b = write_geotiff(str(tmp_path / "b.tif"), [arr], origin=(1.5, 100.0), pixel_size=3.0)  # half-pixel offset
    info_a, info_b = io.get_raster_info(path_a), io.get_raster_info(path_b)
    assert not io.grids_aligned(info_a, info_b)


def test_grids_aligned_false_for_different_pixel_size(tmp_path):
    arr = np.zeros((10, 10), dtype=np.uint16)
    path_a = write_geotiff(str(tmp_path / "a.tif"), [arr], pixel_size=3.0)
    path_b = write_geotiff(str(tmp_path / "b.tif"), [arr], pixel_size=5.0)
    info_a, info_b = io.get_raster_info(path_a), io.get_raster_info(path_b)
    assert not io.grids_aligned(info_a, info_b)


def test_overlap_window_computes_correct_intersection(tmp_path):
    # a: 10x10 at origin (0, 100), pixel 3m -> covers x in [0,30), y in [70,100)
    # b: 10x10 at origin (15, 91), pixel 3m -> covers x in [15,45), y in [61,91)
    # overlap in map coords: x in [15,30), y in [70,91) -> 5x7 px in a's frame? let's just check non-None + sane sizes
    arr = np.zeros((10, 10), dtype=np.uint16)
    path_a = write_geotiff(str(tmp_path / "a.tif"), [arr], origin=(0.0, 100.0), pixel_size=3.0)
    path_b = write_geotiff(str(tmp_path / "b.tif"), [arr], origin=(15.0, 91.0), pixel_size=3.0)
    info_a, info_b = io.get_raster_info(path_a), io.get_raster_info(path_b)

    result = io.overlap_window(info_a, info_b)
    assert result is not None
    window_a, window_b = result
    assert window_a[2] > 0 and window_a[3] > 0
    assert window_a[2] == window_b[2] and window_a[3] == window_b[3]  # same overlap size in both frames
    # a's window should start at pixel (5, 3): x offset (15-0)/3=5, y offset (100-91)/3=3
    assert window_a == (5, 3, 5, 7)
    # b's window should start at pixel (0, 0) since overlap starts exactly at b's origin
    assert window_b == (0, 0, 5, 7)


def test_overlap_window_none_when_disjoint(tmp_path):
    arr = np.zeros((10, 10), dtype=np.uint16)
    path_a = write_geotiff(str(tmp_path / "a.tif"), [arr], origin=(0.0, 100.0), pixel_size=3.0)
    path_b = write_geotiff(str(tmp_path / "b.tif"), [arr], origin=(1000.0, 100.0), pixel_size=3.0)
    info_a, info_b = io.get_raster_info(path_a), io.get_raster_info(path_b)
    assert io.overlap_window(info_a, info_b) is None


def test_windowed_raster_info_shifts_origin_and_size(tmp_path):
    arr = np.zeros((10, 10), dtype=np.uint16)
    path = write_geotiff(str(tmp_path / "a.tif"), [arr], origin=(0.0, 100.0), pixel_size=3.0)
    info = io.get_raster_info(path)

    # window starting at pixel (2, 4) within the 10x10 raster, 5x3 in size
    window = (2, 4, 5, 3)
    windowed = io.windowed_raster_info(info, window)

    assert windowed.width == 5
    assert windowed.height == 3
    # origin shifted by (xoff * pixel_width, yoff * pixel_height): x: 0+2*3=6, y: 100+4*(-3)=88
    assert windowed.geotransform[0] == pytest.approx(6.0)
    assert windowed.geotransform[3] == pytest.approx(88.0)
    # pixel size/rotation/CRS/nodata unchanged
    assert windowed.geotransform[1] == info.geotransform[1]
    assert windowed.geotransform[5] == info.geotransform[5]
    assert windowed.crs == info.crs
    assert windowed.nodata == info.nodata


def test_write_single_band_raster_with_windowed_info_is_correctly_georeferenced(tmp_path):
    # Simulates the candidate-mask bug: writing a sub-window-sized array
    # must use a geotransform shifted to the window's own origin, not the
    # parent raster's, or the written pixels land in the wrong place.
    arr = np.zeros((10, 10), dtype=np.uint16)
    parent_path = write_geotiff(str(tmp_path / "parent.tif"), [arr], origin=(0.0, 100.0), pixel_size=3.0)
    parent_info = io.get_raster_info(parent_path)

    window = (2, 4, 5, 3)  # xoff, yoff, xsize, ysize
    sub_array = np.ones((3, 5), dtype=np.uint8)
    out_path = str(tmp_path / "sub.tif")
    io.write_single_band_raster(sub_array, io.windowed_raster_info(parent_info, window), out_path, gdal.GDT_Byte)

    sub_info = io.get_raster_info(out_path)
    assert sub_info.width == 5 and sub_info.height == 3
    assert sub_info.geotransform[0] == pytest.approx(6.0)   # matches window's true map-space origin
    assert sub_info.geotransform[3] == pytest.approx(88.0)


def test_iter_row_blocks_covers_full_height():
    blocks = list(io.iter_row_blocks(1000, 300))
    assert blocks == [(0, 300), (300, 300), (600, 300), (900, 100)]


def test_nearest_resize_blows_up_and_shrinks():
    small = np.array([[1, 2], [3, 4]])
    big = io.nearest_resize(small, (4, 4))
    assert big.shape == (4, 4)
    # each source pixel replicated into a 2x2 block
    assert (big[:2, :2] == 1).all()
    assert (big[:2, 2:] == 2).all()
    assert (big[2:, :2] == 3).all()
    assert (big[2:, 2:] == 4).all()

    back = io.nearest_resize(big, (2, 2))
    np.testing.assert_array_equal(back, small)


def test_read_block_flat_downsampled_matches_native_shape_and_averages(tmp_path):
    # 4x4, two bands, band 0 has a sharp checkerboard (noise-like); band 1
    # is uniform. downsample_factor=2 should average each 2x2 cell.
    band0 = np.array([[0, 100, 0, 100],
                       [100, 0, 100, 0],
                       [0, 100, 0, 100],
                       [100, 0, 100, 0]], dtype=np.uint16)
    band1 = np.full((4, 4), 50, dtype=np.uint16)
    path = write_geotiff(str(tmp_path / "a.tif"), [band0, band1], nodata=None)

    dataset, bands = io.open_bands(path, [1, 2])
    native = io.read_block_flat(bands, 0, 0, 4, 4)
    downsampled = io.read_block_flat(bands, 0, 0, 4, 4, downsample_factor=2)
    dataset = None

    assert native.shape == downsampled.shape == (16, 2)
    # band 0's checkerboard averages to 50 everywhere once downsampled
    np.testing.assert_allclose(downsampled[:, 0], 50.0)
    # uniform band is unaffected by averaging
    np.testing.assert_allclose(downsampled[:, 1], 50.0)
    # native (undownsampled) read keeps the sharp checkerboard -- not all 50
    assert not np.allclose(native[:, 0], 50.0)


def test_read_block_flat_shared_phase_gives_same_local_grouping_across_offsets(tmp_path):
    # Two windows starting at different xoff/yoff (as reference and target
    # windows onto different rasters would) but reading with the SAME
    # explicit x_phase/y_phase must group local columns/rows into coarse
    # cells at identical *local* boundaries -- regardless of where each
    # window happens to start in its own raster. This is the invariant
    # irmad._load_window_cache and normalize.fit_regression_from_mask rely
    # on to keep every target scene's coarse grid aligned to the reference.
    # Both windows are kept away from the raster edge (xoff >= x_phase) so
    # neither read is edge-clipped -- clipped boundary cells are a separate,
    # already-documented approximation (see the edge-clip test below), not
    # what this test is checking.
    width, height = 20, 1
    row = np.arange(width, dtype=np.float64).reshape(1, width)
    band = np.tile(row, (height, 1)).astype(np.uint16)
    path = write_geotiff(str(tmp_path / "a.tif"), [band], nodata=None)
    dataset, bands = io.open_bands(path, [1])

    factor = 3
    phase = 1  # deliberately not equal to either window's own xoff % factor

    window_a = io.read_block_flat(bands, 4, 0, 9, 1, factor, phase, 0)
    window_b = io.read_block_flat(bands, 7, 0, 9, 1, factor, phase, 0)
    dataset = None

    col_a = window_a[:, 0]
    col_b = window_b[:, 0]

    # group(j) = floor((phase + j) / factor): boundaries fall after local
    # indices 1, 4, 7 -- identical for both windows despite different xoff.
    for j in range(len(col_a) - 1):
        same_group = (phase + j) // factor == (phase + j + 1) // factor
        assert (col_a[j] == col_a[j + 1]) == same_group
        assert (col_b[j] == col_b[j + 1]) == same_group


def test_read_block_flat_handles_phase_pushing_offset_negative(tmp_path):
    # A shared phase inherited from another window's own offset (see the
    # test above / the docstring) can be larger than this window's xoff,
    # driving read_xoff/read_yoff negative -- reproduces a real bug hit in
    # production ("Access window out of range in RasterIO()") where the
    # target window's xoff was smaller than the phase borrowed from the
    # reference window. Must not raise, and must still return the requested
    # shape.
    band = np.arange(100, dtype=np.uint16).reshape(10, 10)
    path = write_geotiff(str(tmp_path / "a.tif"), [band], nodata=None)
    dataset, bands = io.open_bands(path, [1])

    tile = io.read_block_flat(bands, 1, 1, 5, 5, downsample_factor=3, x_phase=2, y_phase=2)
    dataset = None

    assert tile.shape == (25, 1)
    assert np.isfinite(tile).all()


def test_discover_scenes_pairs_files_and_skips_appledouble(tmp_path):
    arr = np.zeros((5, 5), dtype=np.uint16)
    scene_id = "20241005_160142_87_24d6"
    write_geotiff(str(tmp_path / f"{scene_id}_3B_AnalyticMS_SR_harmonized_clip.tif"), [arr])
    write_geotiff(str(tmp_path / f"{scene_id}_3B_udm2_clip.tif"), [arr])
    (tmp_path / f"{scene_id}_metadata.json").write_text(json.dumps({"properties": {"clear_percent": 90}}))
    # AppleDouble junk that matches the same glob suffix -- must be skipped
    (tmp_path / f"._{scene_id}_3B_AnalyticMS_SR_harmonized_clip.tif").write_bytes(b"junk")

    scenes = io.discover_scenes(str(tmp_path))
    assert len(scenes) == 1
    scene = scenes[0]
    assert scene.scene_id == scene_id
    assert scene.udm2_path is not None
    assert scene.metadata_path is not None
    assert scene.acquired == datetime(2024, 10, 5, 16, 1, 42)


def test_discover_scenes_finds_4_and_8_band_products_and_searches_subfolders(tmp_path):
    arr = np.zeros((5, 5), dtype=np.uint16)
    four = tmp_path / "site4b" / "site4b_2021_1_31"
    eight = tmp_path / "site8b" / "site8b_2021_1_31"
    four.mkdir(parents=True)
    eight.mkdir(parents=True)
    write_geotiff(str(four / "20210105_150000_1003_3B_AnalyticMS_SR_harmonized_clip.tif"), [arr])
    write_geotiff(str(eight / "20210102_150000_24a1_3B_AnalyticMS_SR_8b_harmonized_clip.tif"), [arr] * 8)
    write_geotiff(str(eight / "20210102_150000_24a1_3B_udm2_clip.tif"), [arr])
    (eight / "20210102_150000_24a1_metadata.json").write_text("{}")
    (eight / "20210102_150000_24a1_3B_AnalyticMS_SR_8b_harmonized_clip.tif.aux.xml").write_text("<x/>")

    assert io.discover_scenes(str(tmp_path)) == []  # not recursive by default
    scenes = io.discover_scenes(str(tmp_path), recursive=True)
    assert [s.scene_id for s in scenes] == ["20210102_150000_24a1", "20210105_150000_1003"]
    assert scenes[0].analytic_path.endswith("_8b_harmonized_clip.tif")
    assert scenes[0].udm2_path is not None and scenes[0].metadata_path is not None
    assert [s.scene_id for s in io.discover_scenes(str(eight))] == ["20210102_150000_24a1"]
    assert [s.scene_id for s in io.discover_scenes(str(eight), analytic_suffix=io.FOUR_BAND_SUFFIX)] == []

    found = io.discover_scenes_from_ids(str(eight), {"20210102_150000_24a1", "20210101_000000_none"})
    assert [s.analytic_path for s in found] == [scenes[0].analytic_path]


def test_discover_scenes_rejects_a_scene_id_found_twice(tmp_path):
    arr = np.zeros((5, 5), dtype=np.uint16)
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        write_geotiff(str(tmp_path / sub / "20210105_150000_1003_3B_AnalyticMS_SR_harmonized_clip.tif"), [arr])
    with pytest.raises(ValueError, match="found twice"):
        io.discover_scenes(str(tmp_path), recursive=True)
