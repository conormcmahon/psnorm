"""Shared helpers for building small synthetic GeoTIFFs in tests."""

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def write_geotiff(
    path: str,
    bands: list[np.ndarray],
    *,
    origin=(0.0, 100.0),
    pixel_size=3.0,
    band_descriptions: list[str] | None = None,
    nodata: float | None = 0,
    dtype=gdal.GDT_UInt16,
):
    height, width = bands[0].shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, width, height, len(bands), dtype)
    dataset.SetGeoTransform((origin[0], pixel_size, 0.0, origin[1], 0.0, -pixel_size))
    dataset.SetProjection(
        'PROJCS["WGS 84 / UTM zone 18N",GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
        'UNIT["metre",1],AUTHORITY["EPSG","32618"]]'
    )
    for i, arr in enumerate(bands, start=1):
        band = dataset.GetRasterBand(i)
        band.WriteArray(arr)
        if nodata is not None:
            band.SetNoDataValue(nodata)
        if band_descriptions:
            band.SetDescription(band_descriptions[i - 1])
    dataset.FlushCache()
    dataset = None
    return path
