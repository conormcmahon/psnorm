"""Composite output layout: writing in build_composite's block pattern must
not leave orphaned (rewritten) compressed blocks in the file."""

import os

import numpy as np
import pytest
from osgeo import gdal, gdal_array

from psnorm import compositing, io

gdal.UseExceptions()


def _referenced_bytes(path):
    ds = gdal.Open(path)
    total = 0
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b)
        bx, by = band.GetBlockSize()
        for y in range(-(-ds.RasterYSize // by)):
            for x in range(-(-ds.RasterXSize // bx)):
                size = band.GetMetadataItem(f"BLOCK_SIZE_{x}_{y}", "TIFF")
                total += int(size) if size else 0
    return total


def test_output_tile_size_divides_write_blocks():
    assert compositing.output_tile_size(256, 1024) == 256
    assert compositing.output_tile_size(512, 2048) == 512
    with pytest.raises(ValueError):
        compositing.output_tile_size(250, 1000)


def test_block_pattern_writes_leave_no_dead_space(tmp_path):
    block_rows, block_cols = 256, 512
    width, height, n_bands = 1100, 600, 6  # deliberately not multiples of the blocks
    info = io.RasterInfo(path="", width=width, height=height, band_count=n_bands, crs="",
                          geotransform=(0.0, 3.0, 0.0, 0.0, 0.0, -3.0), nodata=None)
    path = str(tmp_path / "composite.tif")
    ds = compositing.create_output(path, info, [f"b{i}" for i in range(n_bands)],
                                   compositing.output_tile_size(block_rows, block_cols))
    rng = np.random.default_rng(0)
    for row_start, n_rows in io.iter_row_blocks(height, block_rows):
        for col_start, n_cols in io.iter_row_blocks(width, block_cols):
            tile = rng.normal(size=(n_bands, n_rows, n_cols)).astype(np.float32)
            gdal_array.DatasetWriteArray(ds, tile, xoff=col_start, yoff=row_start)
            # Real composites are far larger than GDAL's block cache, so
            # blocks get flushed between writes; force that here.
            ds.FlushCache()
    ds = None

    check = gdal.Open(path)
    assert check.GetRasterBand(1).GetBlockSize() == [256, 256]
    assert check.GetMetadataItem("INTERLEAVE", "IMAGE_STRUCTURE") == "BAND"
    # Everything except the (small) TIFF header/IFDs should be live block data.
    assert os.path.getsize(path) - _referenced_bytes(path) < 64 * 1024
