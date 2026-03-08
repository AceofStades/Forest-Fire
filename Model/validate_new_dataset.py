"""
Validates the merged dataset (final_feature_stack_DYNAMIC_new.nc).
Run from the Model/ directory:  python validate_new_dataset.py
"""

import os
import sys

import numpy as np
import xarray as xr

NEW_PATH = "dataset/final_feature_stack_DYNAMIC_new.nc"
OLD_PATH = "dataset/final_feature_stack_DYNAMIC.nc"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def validate(path, label):
    section(f"Validating: {label}")
    if not os.path.exists(path):
        print(f"  ERROR: file not found at {path}")
        return None

    size_mb = os.path.getsize(path) / 1024**2
    print(f"  File size : {size_mb:.1f} MB")

    ds = xr.open_dataset(path, engine="h5netcdf")

    # 1. Dimensions
    print(f"\n[1] Dimensions  : {dict(ds.dims)}")
    print(f"[2] Variables   : {list(ds.data_vars)}")
    print(f"[3] Coordinates : {list(ds.coords)}")

    # 2. NaN check
    print("\n[4] NaN counts per variable:")
    total_nan = 0
    for var in ds.data_vars:
        n = int(ds[var].isnull().sum())
        total_nan += n
        status = "CLEAN" if n == 0 else f"WARNING — {n:,} NaNs"
        print(f"     {var:20s}: {status}")
    print(f"  → Total NaNs: {total_nan:,}")

    # 3. Value ranges
    print("\n[5] Value ranges (min / max / mean):")
    for var in ds.data_vars:
        v = ds[var].values
        mn, mx, mu = np.nanmin(v), np.nanmax(v), np.nanmean(v)
        print(f"     {var:20s}: {mn:10.3f}  {mx:10.3f}  {mu:10.3f}")

    # 4. Binary target check
    print("\n[6] MODIS_FIRE_T1 binary check:")
    if "MODIS_FIRE_T1" in ds.data_vars:
        unique = np.unique(ds["MODIS_FIRE_T1"].values)
        print(f"     Unique values : {unique}")
        fire_pct = float(ds["MODIS_FIRE_T1"].mean()) * 100
        print(f"     Fire pixel %  : {fire_pct:.3f}%")
        if not np.all(np.isin(unique, [0.0, 1.0])):
            print("     CRITICAL: non-binary values detected!")
        else:
            print("     OK — binary (0/1)")
    else:
        print("     WARNING: MODIS_FIRE_T1 not found!")

    # 5. Temporal continuity
    print("\n[7] Temporal continuity:")
    times = ds.valid_time.values
    diffs = np.diff(times) / np.timedelta64(1, "h")
    if np.all(diffs == 1.0):
        print(f"     OK — perfectly hourly ({len(times):,} steps)")
    else:
        gaps = np.where(diffs != 1.0)[0]
        print(f"     WARNING: {len(gaps)} irregular gaps (expected 1 h)")
        for g in gaps[:10]:
            print(f"       idx {g}: gap = {diffs[g]:.1f} h")

    # 6. Spatial extent
    if "latitude" in ds.coords and "longitude" in ds.coords:
        lat = ds.latitude.values
        lon = ds.longitude.values
        print(f"\n[8] Spatial extent:")
        print(f"     Lat : {lat.min():.3f}° – {lat.max():.3f}°  ({len(lat)} pixels)")
        print(f"     Lon : {lon.min():.3f}° – {lon.max():.3f}°  ({len(lon)} pixels)")

    ds.close()
    return {
        "dims": dict(ds.dims),
        "vars": list(ds.data_vars),
        "total_nan": total_nan,
    }


def compare(r_new, r_old):
    section("Comparison: NEW vs OLD DYNAMIC")
    if r_new is None or r_old is None:
        print("  Cannot compare — one file missing.")
        return

    if r_new["dims"] == r_old["dims"]:
        print("  Dimensions   : MATCH ✓")
    else:
        print(f"  Dimensions   : DIFFER")
        print(f"    NEW : {r_new['dims']}")
        print(f"    OLD : {r_old['dims']}")

    new_vars = set(r_new["vars"])
    old_vars = set(r_old["vars"])
    if new_vars == old_vars:
        print(f"  Variables    : MATCH ✓  ({len(new_vars)} vars)")
    else:
        added = new_vars - old_vars
        removed = old_vars - new_vars
        if added:
            print(f"  Added vars   : {sorted(added)}")
        if removed:
            print(f"  Removed vars : {sorted(removed)}")

    if r_new["total_nan"] == 0:
        print("  NaN status   : NEW dataset is clean ✓")
    else:
        print(f"  NaN status   : NEW has {r_new['total_nan']:,} NaNs !")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    r_new = validate(NEW_PATH, "DYNAMIC_new (merged)")
    r_old = validate(OLD_PATH, "DYNAMIC (original)")
    compare(r_new, r_old)

    section("Summary")
    if r_new is not None and r_new["total_nan"] == 0:
        print("  ✅  DYNAMIC_new.nc looks valid — safe to use for training.")
    else:
        print("  ⚠️   Issues detected — review output above before training.")
