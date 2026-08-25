"""
Regression tests for the two georeferencing fixes.

  #2  era5-resample.py   : from_origin() was given cell centres instead of the
                           top-left outer corner -> weather shifted ~5 km SE.
  #4  merge_dynamic.py   : searchsorted() insertion index instead of nearest
                           cell -> every fire point shifted ~half a cell.

Neither test needs the pipeline to be re-run: they replay the old and new code
paths on the real source data and compare against an independent reference.

Run from the Model/ directory:
    ../.venv/bin/python dataset-validation-scripts/test_preprocessing_fixes.py
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.warp import Resampling, reproject

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing", "resampling"))

from merge_dynamic import nearest_index  # noqa: E402

PUB = "dataset/final_feature_stack_DYNAMIC_interpolated.nc"
ERA5_SRC = "dataset/ERA5-Land/final-era5_rechunked.nc"
MODIS_CSV = "dataset/MODIS/final-modis.csv"

TARGET_BOUNDS = (77.5, 28.7, 81.1, 31.5)  # left, bottom, right, top
TARGET_W, TARGET_H = 400, 311

_failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def hdr(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


# ---------------------------------------------------------------- fix #4 ----

def old_index(grid, points):
    """The previous searchsorted-based mapping, verbatim."""
    if grid[1] > grid[0]:
        idx = np.searchsorted(grid, points)
    else:
        idx = len(grid) - 1 - np.searchsorted(grid[::-1], points)
    return np.clip(idx, 0, len(grid) - 1)


def brute_nearest(grid, points):
    return np.abs(grid[None, :] - np.asarray(points)[:, None]).argmin(axis=1)


def test_nearest_index():
    hdr("Fix #4 - nearest_index() in merge_dynamic.py")

    rng = np.random.default_rng(0)

    # Synthetic grids, both orientations, plus out-of-range points.
    for name, grid in [
        ("ascending", np.linspace(28.7, 31.5, 311)),
        ("descending", np.linspace(31.5, 28.7, 311)),
        ("ascending lon", np.linspace(77.5, 81.1, 400)),
        ("descending lon", np.linspace(81.1, 77.5, 400)),
    ]:
        pts = rng.uniform(grid.min() - 0.5, grid.max() + 0.5, 20000)
        got = nearest_index(grid, pts)
        want = brute_nearest(grid, pts)
        check(f"{name}: matches brute-force argmin on 20k pts",
              np.array_equal(got, want),
              f"mismatches={int((got != want).sum())}")

    # Exact cell centres must map to themselves.
    grid = np.linspace(31.5, 28.7, 311)
    got = nearest_index(grid, grid)
    check("exact cell centres map to their own index",
          np.array_equal(got, np.arange(len(grid))))

    # Out-of-range clamps to the edge cell (previous behaviour preserved).
    got = nearest_index(grid, np.array([99.0, -99.0]))
    check("out-of-range points clamp to edge cells",
          got[0] == 0 and got[1] == len(grid) - 1, f"got={got.tolist()}")

    # Midpoint between two centres: must pick one of the two neighbours.
    mid = (grid[10] + grid[11]) / 2
    got = int(nearest_index(grid, np.array([mid]))[0])
    check("midpoint resolves to an adjacent cell", got in (10, 11), f"got={got}")

    # Real grid + real MODIS points: quantify the improvement.
    if os.path.exists(PUB) and os.path.exists(MODIS_CSV):
        ds = xr.open_dataset(PUB, engine="h5netcdf")
        lats, lons = ds.latitude.values, ds.longitude.values
        df = pd.read_csv(MODIS_CSV)
        p_lat, p_lon = df["latitude"].values, df["longitude"].values

        old_la, old_lo = old_index(lats, p_lat), old_index(lons, p_lon)
        new_la, new_lo = nearest_index(lats, p_lat), nearest_index(lons, p_lon)
        ref_la, ref_lo = brute_nearest(lats, p_lat), brute_nearest(lons, p_lon)

        old_wrong = ((old_la != ref_la) | (old_lo != ref_lo)).mean()
        new_wrong = ((new_la != ref_la) | (new_lo != ref_lo)).mean()
        print(f"\n  real MODIS detections: {len(df)}")
        print(f"    old code, wrong cell : {100 * old_wrong:.1f}%")
        print(f"    new code, wrong cell : {100 * new_wrong:.1f}%")
        print(f"    old signed bias      : lat {(old_la - ref_la).mean():+.3f}, "
              f"lon {(old_lo - ref_lo).mean():+.3f} cells")
        print(f"    new signed bias      : lat {(new_la - ref_la).mean():+.3f}, "
              f"lon {(new_lo - ref_lo).mean():+.3f} cells")
        check("real detections all land in the true nearest cell", new_wrong == 0.0)


# ---------------------------------------------------------------- fix #2 ----

def _resample_t2m(use_fix):
    """Replay era5-resample.py's per-timestep reproject, old or new transform."""
    src = xr.open_dataset(ERA5_SRC, engine="h5netcdf")
    chunk = src["t2m"].isel(valid_time=0)
    lons = chunk.longitude.values
    lats = chunk.latitude.values
    data = chunk.values[None, :, :].astype(np.float32)

    x_res = abs(lons[1] - lons[0])
    y_res = abs(lats[1] - lats[0])

    if use_fix:
        west = lons.min() - x_res / 2
        north = lats.max() + y_res / 2
        if lats[1] > lats[0]:
            data = data[:, ::-1, :]
    else:
        west = lons[0]
        north = lats[0]

    tr = rasterio.transform.from_origin(west, north, x_res, y_res)
    out = np.empty((1, TARGET_H, TARGET_W), dtype=np.float32)
    reproject(
        source=data,
        destination=out,
        src_transform=tr,
        src_crs="EPSG:4326",
        dst_transform=rasterio.transform.from_bounds(*TARGET_BOUNDS, TARGET_W, TARGET_H),
        dst_crs="EPSG:4326",
        resampling=Resampling.bilinear,
    )
    return out[0]


def _reference_t2m():
    """Independent ground truth: interpolate ERA5 grid points onto target centres."""
    src = xr.open_dataset(ERA5_SRC, engine="h5netcdf")
    left, bottom, right, top = TARGET_BOUNDS
    dlat = (top - bottom) / TARGET_H
    dlon = (right - left) / TARGET_W
    clat = top - (np.arange(TARGET_H) + 0.5) * dlat
    clon = left + (np.arange(TARGET_W) + 0.5) * dlon
    return (src["t2m"].isel(valid_time=0)
            .interp(latitude=("latitude", clat), longitude=("longitude", clon),
                    method="linear").values)


def _best_shift(obs, ref, rng=9, margin=15):
    best = None
    m = slice(margin, -margin)
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            o = np.roll(np.roll(obs, dy, axis=0), dx, axis=1)
            r = np.sqrt(np.nanmean((o[m, m] - ref[m, m]) ** 2))
            if best is None or r < best[0]:
                best = (r, dy, dx)
    return best


def test_era5_transform():
    hdr("Fix #2 - from_origin() corner vs centre in era5-resample.py")
    if not os.path.exists(ERA5_SRC):
        print(f"  SKIP: {ERA5_SRC} not found")
        return

    ref = _reference_t2m()
    old = _resample_t2m(use_fix=False)
    new = _resample_t2m(use_fix=True)

    m = slice(15, -15)
    rmse_old = np.sqrt(np.nanmean((old[m, m] - ref[m, m]) ** 2))
    rmse_new = np.sqrt(np.nanmean((new[m, m] - ref[m, m]) ** 2))
    r_old, dy_old, dx_old = _best_shift(old, ref)
    r_new, dy_new, dx_new = _best_shift(new, ref)

    print(f"  old transform: RMSE {rmse_old:.4f} K, best shift dy={dy_old:+d} dx={dx_old:+d}")
    print(f"  new transform: RMSE {rmse_new:.4f} K, best shift dy={dy_new:+d} dx={dx_new:+d}")

    check("old code reproduces the ~5 px displacement seen in the published file",
          (dy_old, dx_old) != (0, 0), f"shift=({dy_old:+d},{dx_old:+d})")
    check("new code needs no shift to match the reference",
          (dy_new, dx_new) == (0, 0), f"shift=({dy_new:+d},{dx_new:+d})")
    check("new code is closer to the reference than old",
          rmse_new < rmse_old, f"{rmse_new:.4f} K < {rmse_old:.4f} K")

    # Guard the flip branch: an ascending-latitude source must not come out upside down.
    src = xr.open_dataset(ERA5_SRC, engine="h5netcdf")
    lats = src.latitude.values
    print(f"\n  source latitude is {'descending' if lats[1] < lats[0] else 'ascending'} "
          f"({lats[0]:.2f} -> {lats[-1]:.2f}); flip branch "
          f"{'not exercised' if lats[1] < lats[0] else 'exercised'} by this data")


def main():
    if not os.path.exists("dataset"):
        print("Run this from the Model/ directory.")
        return 1
    test_nearest_index()
    test_era5_transform()

    hdr("SUMMARY")
    if _failures:
        print(f"  {len(_failures)} FAILED:")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
