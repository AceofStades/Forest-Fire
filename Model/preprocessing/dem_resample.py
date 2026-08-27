"""Copernicus DEM GLO-30 -> the 1 km target grid.

Replaces the earlier CartoDEM tiles (P5_PAN_CD_*), which are an NRSC/ISRO
product distributed through Bhuvan. Bhuvan's terms grant a single-user,
internal-use licence with digital databases restricted to authorised
government users, so a derived product cannot be redistributed -- the same
reason the Bhuvan LULC layer was replaced by ESA WorldCover.

Copernicus DEM GLO-30 is free for any use with attribution and covers the
whole grid, so the 0-means-nodata sentinel the old DEM needed disappears.

Run from the Model/ directory:
    ../.venv/bin/python preprocessing/dem_resample.py
"""

import glob
import os
import subprocess

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

TILE_GLOB = "dataset/CopernicusDEM/Copernicus_DSM_COG_10_*_DEM.tif"
VRT_PATH = "dataset/CopernicusDEM/cop_dem_mosaic.vrt"
OUT_TIF = "dataset/resampled-fix/dem_copernicus_1km.tif"

WEST, EAST = 77.5, 81.1
SOUTH, NORTH = 28.7, 31.5
WIDTH, HEIGHT = 400, 311

# The source is ~30 m and the target ~1 km, so each output cell averages about
# 1,100 input pixels. Bilinear on its own would point-sample; `average` is the
# honest reduction for a continuous field at this ratio.
RESAMPLING = Resampling.average


def build_vrt():
    tiles = sorted(glob.glob(TILE_GLOB))
    if not tiles:
        raise SystemExit(f"No tiles matched {TILE_GLOB}")
    print(f"Mosaicking {len(tiles)} tiles -> {VRT_PATH}")
    subprocess.run(["gdalbuildvrt", "-q", VRT_PATH, *tiles], check=True)
    return VRT_PATH


def resample(vrt_path):
    transform = rasterio.transform.from_bounds(WEST, SOUTH, EAST, NORTH, WIDTH, HEIGHT)
    dst = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    with rasterio.open(vrt_path) as src:
        print(f"  source: {src.width} x {src.height} @ {src.res[0]:.7f} deg")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=RESAMPLING,
            num_threads=os.cpu_count() or 4,
        )
    return dst, transform


def main():
    dem, transform = resample(build_vrt())

    n_nan = int(np.isnan(dem).sum())
    print(f"\n  cells with no source data: {n_nan:,} ({100 * n_nan / dem.size:.3f}%)")
    if n_nan:
        # Should not happen over land; fill so downstream code never sees NaN.
        dem = np.where(np.isnan(dem), 0.0, dem)

    print(f"  elevation range : {dem.min():.1f} .. {dem.max():.1f} m")
    print(f"  mean elevation  : {dem.mean():.1f} m")

    os.makedirs(os.path.dirname(OUT_TIF), exist_ok=True)
    with rasterio.open(OUT_TIF, "w", driver="GTiff", height=HEIGHT, width=WIDTH,
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=transform, compress="deflate") as out:
        out.write(dem.astype(np.float32), 1)
        out.update_tags(
            source="Copernicus DEM GLO-30 (COP-DEM_GLO-30-DGED / DTED), 2021",
            licence="Free for any use with attribution; (c) DLR e.V. / Airbus DS",
            method="average over ~30 m source pixels",
        )
    print(f"  wrote {OUT_TIF}")

    # Cross-check against the CartoDEM layer being replaced: the two should
    # broadly agree, which is evidence the new one is correctly georeferenced.
    old = "dataset/resampled-fix/dem_resampled.tif"
    if os.path.exists(old):
        with rasterio.open(old) as s:
            prev = s.read(1)
        both = (prev > 0) & (dem > 0)
        diff = dem[both] - prev[both]
        print(f"\n  vs the CartoDEM layer, over {both.sum():,} cells valid in both:")
        print(f"    mean difference   {diff.mean():+.1f} m")
        print(f"    median |difference| {np.median(np.abs(diff)):.1f} m")
        print(f"    correlation       {np.corrcoef(dem[both], prev[both])[0, 1]:.5f}")
        print(f"    cells the old DEM had as nodata but this one resolves: "
              f"{int(((prev == 0) & (dem > 0)).sum()):,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
