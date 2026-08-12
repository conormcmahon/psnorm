"""Chunked application of a fitted NormalizationModel's per-band gain/offset
to a full scene, streamed in row-blocks so no scene is ever fully loaded
into memory regardless of its size."""

from __future__ import annotations

import numpy as np
from osgeo import gdal, gdal_array

from . import io
from .normalize import NormalizationModel

gdal.UseExceptions()

_GDT_RANGES = {
    gdal.GDT_Byte: (0, 255),
    gdal.GDT_UInt16: (0, 65535),
    gdal.GDT_Int16: (-32768, 32767),
    gdal.GDT_UInt32: (0, 4294967295),
    gdal.GDT_Int32: (-2147483648, 2147483647),
}


def _clip_for_dtype(arr: np.ndarray, gdal_dtype: int) -> np.ndarray:
    rng = _GDT_RANGES.get(gdal_dtype)
    if rng is None:
        return arr
    return np.clip(arr, rng[0], rng[1])


def apply_model(
    target_path: str,
    model: NormalizationModel,
    output_path: str,
    band_names: list[str],
    *,
    block_rows: int = io.DEFAULT_BLOCK_ROWS,
    out_dtype: int | None = None,
) -> str:
    """Write `target_path` normalized by `model` (`intercept + slope*x` per
    band) to `output_path`, preserving georeferencing/nodata."""
    src_ds = gdal.Open(target_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise ValueError(f"Could not open raster: {target_path}")
    width, height, band_count = src_ds.RasterXSize, src_ds.RasterYSize, src_ds.RasterCount
    if band_count != len(band_names):
        raise ValueError(
            f"'{target_path}' has {band_count} bands but band_names has "
            f"{len(band_names)} entries."
        )

    if out_dtype is None:
        out_dtype = src_ds.GetRasterBand(1).DataType
    np_dtype = gdal_array.GDALTypeCodeToNumericTypeCode(out_dtype)

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path, width, height, band_count, out_dtype,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())

    src_bands = [src_ds.GetRasterBand(i + 1) for i in range(band_count)]
    out_bands = [out_ds.GetRasterBand(i + 1) for i in range(band_count)]
    band_models = [model.band(name) for name in band_names]

    for src_band, out_band in zip(src_bands, out_bands):
        nodata = src_band.GetNoDataValue()
        if nodata is not None:
            out_band.SetNoDataValue(nodata)

    for row_off, n_rows in io.iter_row_blocks(height, block_rows):
        for src_band, out_band, band_model in zip(src_bands, out_bands, band_models):
            arr = src_band.ReadAsArray(0, row_off, width, n_rows).astype(np.float64)
            nodata = src_band.GetNoDataValue()
            transformed = band_model.intercept + band_model.slope * arr
            transformed = _clip_for_dtype(transformed, out_dtype)
            if nodata is not None:
                transformed = np.where(arr == nodata, nodata, transformed)
            out_band.WriteArray(transformed.astype(np_dtype), 0, row_off)

    for out_band in out_bands:
        out_band.FlushCache()
    out_ds = None
    src_ds = None
    return output_path
