"""ESA WorldCover 10 m -> the 1 km target grid, by majority class.

Replaces the earlier Bhuvan LULC layer, which was reconstructed from a lossy
RGB map render and which NRSC's terms do not permit us to redistribute.
WorldCover is CC-BY-4.0, ships real class codes with names, and covers the
whole grid rectangle rather than a state polygon.

Downsampling is 100x in each direction, so ~10,000 source pixels fall in every
target cell. Nearest-neighbour would keep exactly one of them and throw the
rest away; this uses the MAJORITY class instead, and also records the winning
class's share so callers can tell a clean cell from a coin-flip.

Run from the Model/ directory:
    ../.venv/bin/python preprocessing/worldcover_resample.py
"""

import glob
import os
import subprocess

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

TILE_GLOB = "dataset/WorldCover/ESA_WorldCover_10m_2021_v200_*_Map.tif"
VRT_PATH = "dataset/WorldCover/worldcover_mosaic.vrt"
OUT_TIF = "dataset/resampled-fix/worldcover_1km.tif"
OUT_SHARE = "dataset/resampled-fix/worldcover_1km_purity.tif"
LEGEND_CSV = "dataset/WorldCover/worldcover_legend.csv"

# Target grid: identical to final_feature_stack_RELEASE_v2.nc (outer corners).
WEST, EAST = 77.5, 81.1
SOUTH, NORTH = 28.7, 31.5
WIDTH, HEIGHT = 400, 311

# ESA WorldCover v200 legend. Burnability is our own judgement, not ESA's:
# it marks classes that can carry a vegetation fire.
LEGEND = [
    (10, "Tree cover", True),
    (20, "Shrubland", True),
    (30, "Grassland", True),
    (40, "Cropland", True),
    (50, "Built-up", False),
    (60, "Bare / sparse vegetation", False),
    (70, "Snow and ice", False),
    (80, "Permanent water bodies", False),
    (90, "Herbaceous wetland", True),
    (95, "Mangroves", True),
    (100, "Moss and lichen", True),
]
NODATA = 0


def build_vrt():
    tiles = sorted(glob.glob(TILE_GLOB))
    if not tiles:
        raise SystemExit(f"No tiles matched {TILE_GLOB}")
    print(f"Mosaicking {len(tiles)} tiles -> {VRT_PATH}")
    subprocess.run(["gdalbuildvrt", "-q", VRT_PATH, *tiles], check=True)
    return VRT_PATH


def majority_resample(vrt_path):
    """Majority class per target cell, plus the winning class's share.

    Done by counting each class separately: a class's presence mask is averaged
    down to the target grid, which gives that class's exact areal fraction.
    Taking the argmax over classes gives the true majority, and the max gives
    its share. This is exact, unlike a subsampled mode.
    """
    transform = rasterio.transform.from_bounds(WEST, SOUTH, EAST, NORTH, WIDTH, HEIGHT)
    codes = [c for c, _, _ in LEGEND]
    frac = np.zeros((len(codes), HEIGHT, WIDTH), dtype=np.float32)

    with rasterio.open(vrt_path) as src:
        print(f"  source: {src.width} x {src.height} @ {src.res[0]:.7f} deg")
        with WarpedVRT(src, crs=src.crs, transform=transform,
                       width=WIDTH, height=HEIGHT,
                       resampling=Resampling.nearest) as _probe:
            pass
        # Read only the window covering the target bounds, at full 10 m res.
        win = rasterio.windows.from_bounds(WEST, SOUTH, EAST, NORTH, src.transform)
        win = win.round_offsets().round_lengths()
        print(f"  reading window {int(win.width)} x {int(win.height)} px "
              f"({win.width * win.height / 1e9:.2f} G pixels)")
        block = src.read(1, window=win)

    src_h, src_w = block.shape
    # Integer block factors; trim the remainder so reshape-reduce is exact.
    fy, fx = src_h // HEIGHT, src_w // WIDTH
    print(f"  block factor {fy} x {fx} ({fy * fx:,} source px per target cell)")
    trimmed = block[: HEIGHT * fy, : WIDTH * fx]

    for i, code in enumerate(codes):
        m = (trimmed == code).reshape(HEIGHT, fy, WIDTH, fx)
        frac[i] = m.mean(axis=(1, 3), dtype=np.float32)
        print(f"    class {code:>3} {LEGEND[i][1]:<26} "
              f"{100 * frac[i].mean():6.2f}% of grid area")

    total = frac.sum(axis=0)
    winner = np.argmax(frac, axis=0)
    lulc = np.take(np.array(codes, dtype=np.uint8), winner)
    share = np.max(frac, axis=0)

    # Cells with no classified source at all (outside WorldCover coverage).
    empty = total < 0.5
    lulc[empty] = NODATA
    share[empty] = 0.0
    print(f"\n  cells with no classified source: {empty.sum():,} "
          f"({100 * empty.mean():.3f}%)")
    return lulc, share, transform


def write(lulc, share, transform):
    os.makedirs(os.path.dirname(OUT_TIF), exist_ok=True)
    common = dict(driver="GTiff", height=HEIGHT, width=WIDTH, count=1,
                  crs="EPSG:4326", transform=transform, compress="deflate")
    with rasterio.open(OUT_TIF, "w", dtype="uint8", nodata=NODATA, **common) as dst:
        dst.write(lulc, 1)
        dst.update_tags(source="ESA WorldCover 10m v200 (2021)",
                        licence="CC-BY-4.0",
                        method="areal majority over 10 m source pixels")
    with rasterio.open(OUT_SHARE, "w", dtype="float32", **common) as dst:
        dst.write(share.astype(np.float32), 1)
    print(f"  wrote {OUT_TIF}")
    print(f"  wrote {OUT_SHARE}")

    pd.DataFrame(
        [(c, n, b) for c, n, b in LEGEND] + [(NODATA, "No data", False)],
        columns=["code", "name", "is_burnable"],
    ).sort_values("code").to_csv(LEGEND_CSV, index=False)
    print(f"  wrote {LEGEND_CSV}")


def report(lulc, share):
    names = {c: n for c, n, _ in LEGEND}
    names[NODATA] = "No data"
    print("\n  majority class composition of the 1 km grid:")
    vals, cnts = np.unique(lulc, return_counts=True)
    for v, c in sorted(zip(vals, cnts), key=lambda t: -t[1]):
        print(f"    {v:>3} {names.get(int(v), '?'):<26} {c:>7,} cells "
              f"({100 * c / lulc.size:5.2f}%)")
    print(f"\n  majority-class purity: mean {share.mean():.3f}, "
          f"median {np.median(share):.3f}")
    print(f"  cells where the winner holds <50% of the cell: "
          f"{(share < 0.5).sum():,} ({100 * (share < 0.5).mean():.1f}%)")


def main():
    vrt = build_vrt()
    lulc, share, transform = majority_resample(vrt)
    write(lulc, share, transform)
    report(lulc, share)
    print("\nDone.")


if __name__ == "__main__":
    main()
