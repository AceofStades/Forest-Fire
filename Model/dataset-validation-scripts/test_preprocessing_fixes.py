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


# ---------------------------------------------------------------- fix #5 ----

def test_cell_centre_coords():
    hdr("Fix #5 - output coordinate labels are cell centres")
    left, bottom, right, top = TARGET_BOUNDS

    # The formula now used in era5-resample.py
    cell_lat = (top - bottom) / TARGET_H
    cell_lon = (right - left) / TARGET_W
    lats = top - (np.arange(TARGET_H) + 0.5) * cell_lat
    lons = left + (np.arange(TARGET_W) + 0.5) * cell_lon

    # Ground truth: the centres of the raster reproject actually writes into
    tr = rasterio.transform.from_bounds(*TARGET_BOUNDS, TARGET_W, TARGET_H)
    ref_lons, _ = rasterio.transform.xy(tr, [0] * TARGET_W, list(range(TARGET_W)))
    _, ref_lats = rasterio.transform.xy(tr, list(range(TARGET_H)), [0] * TARGET_H)

    check("latitudes match the raster's true cell centres",
          np.allclose(lats, ref_lats), f"max err {np.abs(lats - ref_lats).max():.3e}")
    check("longitudes match the raster's true cell centres",
          np.allclose(lons, ref_lons), f"max err {np.abs(lons - ref_lons).max():.3e}")
    check("spacing is extent/N, not extent/(N-1)",
          np.isclose(abs(lats[1] - lats[0]), cell_lat))
    check("no coordinate falls outside the declared bounds",
          lats.max() < top and lats.min() > bottom
          and lons.max() < right and lons.min() > left)

    old = np.linspace(top, bottom, TARGET_H)
    drift = abs(abs(old[1] - old[0]) - cell_lat) * TARGET_H
    print(f"\n  old linspace drift across the grid: {drift:.6f} deg "
          f"({drift * 111:.3f} km ~ {drift / cell_lat:.2f} cells)")
    print(f"  new formula drift: {np.abs(lats - ref_lats).max() * 111:.6f} km")


# ------------------------------------------------------- fixes #9 and #12 ----

def test_fire_mask_filtering(tmpdir="/tmp/claude-1000"):
    hdr("Fixes #9 / #8 - out-of-domain and low-confidence detections are dropped")
    import merge_dynamic as md

    lats = np.linspace(31.0, 30.0, 50)      # descending, like the real grid
    lons = np.linspace(78.0, 79.0, 60)
    times = pd.date_range("2016-04-01", periods=6, freq="h")
    template = xr.Dataset(coords={"valid_time": times,
                                  "latitude": lats, "longitude": lons})

    rows = [
        # in-domain, high confidence -> must be kept.
        # Deliberately off cell boundaries: a point exactly between two centres
        # is a genuine tie, and tie handling is covered by test_nearest_index.
        (30.52, 78.53, 90, 0, "2016-04-01 01:00:00"),
        (30.9, 78.1, 75, 0, "2016-04-01 02:00:00"),
        # out of the spatial domain -> must be dropped, NOT clamped to the border
        (45.0, 78.5, 95, 0, "2016-04-01 01:00:00"),
        (30.5, 20.0, 95, 0, "2016-04-01 01:00:00"),
        # below the confidence floor -> dropped
        (30.4, 78.4, 5, 0, "2016-04-01 03:00:00"),
        # non-vegetation source -> dropped
        (30.6, 78.6, 99, 2, "2016-04-01 04:00:00"),
    ]
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "confidence",
                                     "type", "acq_timestamp"])
    os.makedirs(tmpdir, exist_ok=True)
    csv = os.path.join(tmpdir, "_test_modis.csv")
    df.to_csv(csv, index=False)

    out = md.generate_dynamic_fire_mask(csv, template)
    observed = out["OBSERVED_FIRE"].values
    active = out["ACTIVE_FIRE"].values
    burned = out["BURNED_AREA"].values

    check("exactly the 2 valid detections are rasterised",
          observed.sum() == 2, f"got {observed.sum():.0f}")

    border = np.zeros(observed.shape[1:], dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    check("no fire manufactured on the border by clamping",
          observed[:, border].sum() == 0,
          f"border fire = {observed[:, border].sum():.0f}")

    # The kept detections must land in their true nearest cell.
    for lat, lon, t_idx in [(30.52, 78.53, 1), (30.9, 78.1, 2)]:
        yi = int(np.abs(lats - lat).argmin())
        xi = int(np.abs(lons - lon).argmin())
        check(f"detection ({lat}, {lon}) marked at nearest cell ({yi}, {xi})",
              observed[t_idx, yi, xi] == 1.0)

    os.remove(csv)


def test_fire_channel_semantics(tmpdir="/tmp/claude-1000"):
    hdr("Fix #3 - ACTIVE_FIRE is causal and extinguishes; BURNED_AREA is monotone")
    import merge_dynamic as md

    lats = np.linspace(31.0, 30.0, 20)
    lons = np.linspace(78.0, 79.0, 20)
    times = pd.date_range("2016-04-01", periods=48, freq="h")
    template = xr.Dataset(coords={"valid_time": times,
                                  "latitude": lats, "longitude": lons})

    # One detection at hour 10, another elsewhere at hour 30.
    df = pd.DataFrame(
        [(30.52, 78.53, 90, 0, "2016-04-01 10:00:00"),
         (30.72, 78.23, 90, 0, "2016-04-01 06:00:00")],
        columns=["latitude", "longitude", "confidence", "type", "acq_timestamp"])
    os.makedirs(tmpdir, exist_ok=True)
    csv = os.path.join(tmpdir, "_test_modis_sem.csv")
    df.to_csv(csv, index=False)

    out = md.generate_dynamic_fire_mask(csv, template)
    observed = out["OBSERVED_FIRE"].values
    active = out["ACTIVE_FIRE"].values
    burned = out["BURNED_AREA"].values
    P = md.PERSISTENCE_HOURS
    os.remove(csv)

    # Causality: no cell may be alight before its own first observation.
    first_obs = np.where(observed.any(axis=(1, 2)))[0].min()
    check("nothing is alight before the first observation",
          active[:first_obs].sum() == 0,
          f"first obs at t={first_obs}, prior fire={active[:first_obs].sum():.0f}")

    for t in range(active.shape[0]):
        lit = active[t] > 0
        if not lit.any():
            continue
        # every lit cell must have been observed at some t' <= t
        prior = observed[:t + 1].any(axis=0)
        if not np.all(prior[lit]):
            check(f"frame {t} contains fire with no prior observation", False)
            break
    else:
        check("every lit cell traces back to an earlier or current observation", True)

    # Bounded persistence: each detection burns for exactly P hours.
    yi = int(np.abs(lats - 30.52).argmin())
    xi = int(np.abs(lons - 78.53).argmin())
    run = active[:, yi, xi]
    check(f"detection at t=10 burns for exactly {P} h then goes out",
          run[10:10 + P].all() and run[:10].sum() == 0
          and run[10 + P:].sum() == 0,
          f"run={np.where(run > 0)[0].tolist()}")

    # Extinction actually happens somewhere in the series.
    per_frame = active.sum(axis=(1, 2))
    check("total active fire decreases at some point (fires go out)",
          (np.diff(per_frame) < 0).any(),
          f"decreases at {(np.diff(per_frame) < 0).sum()} steps")

    # BURNED_AREA is monotone and is the running max of ACTIVE_FIRE.
    check("BURNED_AREA never decreases",
          bool((np.diff(burned, axis=0) >= 0).all()))
    check("BURNED_AREA equals the running max of ACTIVE_FIRE",
          np.array_equal(burned, np.maximum.accumulate(active, axis=0)))
    check("ACTIVE_FIRE is a subset of BURNED_AREA",
          bool((active <= burned).all()))
    check("OBSERVED_FIRE is a subset of ACTIVE_FIRE",
          bool((observed <= active).all()))


def test_lulc_classification():
    hdr("Fix #1 - land cover holds real, named class codes")
    tif = "dataset/resampled-fix/worldcover_1km.tif"
    legend = "dataset/WorldCover/worldcover_legend.csv"
    if not os.path.exists(tif):
        print("  SKIP: run preprocessing/worldcover_resample.py first")
        return

    import csv as _csv
    with rasterio.open(tif) as s:
        lulc = s.read(1)
        check("land cover raster is single-band uint8",
              s.count == 1 and s.dtypes[0] == "uint8",
              f"count={s.count} dtype={s.dtypes[0]}")
        check("raster sits on the target grid",
              (s.height, s.width) == (TARGET_H, TARGET_W),
              f"{s.height}x{s.width}")
        b = s.bounds
        check("raster bounds match the target bounds",
              max(abs(a - t) for a, t in zip(
                  (b.left, b.bottom, b.right, b.top), TARGET_BOUNDS)) < 1e-9,
              f"{tuple(round(v, 6) for v in b)}")

    with open(legend) as f:
        rows = list(_csv.DictReader(f))
    codes = {int(r["code"]) for r in rows}
    names = {int(r["code"]): r["name"] for r in rows}
    present = set(np.unique(lulc).tolist())

    check("every code in the raster appears in the legend",
          present <= codes, f"orphans={sorted(present - codes)}")
    check("every present code has a real class name",
          all(names.get(c, "").strip() not in ("", "?") for c in present),
          "the Bhuvan render this replaced had no recoverable names")
    check("codes are ESA WorldCover codes",
          present <= {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100},
          f"unexpected={sorted(present - {0,10,20,30,40,50,60,70,80,90,95,100})}")

    # WorldCover is global, so the off-map background that dominated the Bhuvan
    # layer (57% of the grid) should be entirely absent.
    nodata_share = float((lulc == 0).mean())
    check("no off-map background remains", nodata_share == 0.0,
          f"{100 * nodata_share:.3f}% unclassified")

    burnable = {int(r["code"]) for r in rows
                if r["is_burnable"].strip().lower() == "true"}
    share = float(np.isin(lulc, list(burnable)).mean())
    print(f"\n  {len(present)} distinct classes on the 1 km grid, "
          f"{100 * share:.1f}% burnable")


def test_metadata():
    hdr("Fix #12 - published file is self-describing")
    import merge_dynamic as md

    lats = np.linspace(31.0, 30.0, 5)
    lons = np.linspace(78.0, 79.0, 6)
    times = pd.date_range("2016-04-01", periods=3, freq="h")
    ds = xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"),
                    np.zeros((3, 5, 6), dtype=np.float32)),
            "DEM": (("latitude", "longitude"), np.zeros((5, 6), dtype=np.float32)),
            "OBSERVED_FIRE": (("valid_time", "latitude", "longitude"),
                              np.zeros((3, 5, 6), dtype=np.float32)),
            "ACTIVE_FIRE": (("valid_time", "latitude", "longitude"),
                            np.zeros((3, 5, 6), dtype=np.float32)),
            "BURNED_AREA": (("valid_time", "latitude", "longitude"),
                            np.zeros((3, 5, 6), dtype=np.float32)),
        },
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    out = md.add_metadata(ds)

    check("global title/summary present", bool(out.attrs.get("title"))
          and bool(out.attrs.get("summary")))
    check("CRS recorded", out.attrs.get("spatial_ref") == "EPSG:4326")
    check("coordinate convention stated",
          "centre" in out.attrs.get("coordinate_convention", ""))
    check("confidence threshold recorded",
          out.attrs.get("modis_min_confidence") == md.MIN_CONFIDENCE)
    check("every data variable has units",
          all(out[v].attrs.get("units") for v in out.data_vars),
          str({v: out[v].attrs.get("units") for v in out.data_vars}))
    check("lat/lon carry CF standard names",
          out.latitude.attrs.get("standard_name") == "latitude"
          and out.longitude.attrs.get("standard_name") == "longitude")


def main():
    if not os.path.exists("dataset"):
        print("Run this from the Model/ directory.")
        return 1
    test_nearest_index()
    test_era5_transform()
    test_cell_centre_coords()
    test_fire_mask_filtering()
    test_fire_channel_semantics()
    test_lulc_classification()
    test_metadata()

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
