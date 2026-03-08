"""
Check whether MODIS_FIRE_T1 was copied statically (same frame repeated)
or actually varies over time in each dataset.
Run from the repo root:  python Model/check_modis_temporal.py
"""
import os
import sys

import numpy as np
import xarray as xr

DATASETS = [
    ("DYNAMIC_new", "Model/dataset/final_feature_stack_DYNAMIC_new.nc"),
    ("DYNAMIC_old", "Model/dataset/final_feature_stack_DYNAMIC.nc"),
    ("MASTER",      "Model/dataset/final_feature_stack_MASTER.nc"),
]

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

for label, path in DATASETS:
    if not os.path.exists(path):
        print(f"\n=== {label} === MISSING: {path}")
        continue

    print(f"\n=== {label} ===")
    ds = xr.open_dataset(path, engine="h5netcdf")
    fire = ds["MODIS_FIRE_T1"].values.astype(np.float32)   # (T, H, W)
    ds.close()

    T = fire.shape[0]

    # How many time steps have ANY fire pixel
    fire_per_frame = fire.sum(axis=(1, 2))
    frames_with_fire = int((fire_per_frame > 0).sum())

    # Frame-to-frame changes (transition count)
    diffs = np.diff(fire, axis=0)                          # (T-1, H, W)
    changed = (np.abs(diffs).sum(axis=(1, 2)) > 0)
    frames_that_changed = int(changed.sum())

    # Mean temporal variance per pixel
    temporal_var = float(fire.var(axis=0).mean())

    # Are ALL frames literally identical to frame 0?
    all_same = bool(np.all(fire == fire[0:1, :, :]))

    # How many unique fire totals (coarse uniqueness proxy)
    unique_totals = len(np.unique(fire_per_frame))

    print(f"  Timesteps              : {T}")
    print(f"  Frames with any fire   : {frames_with_fire} / {T}  ({100*frames_with_fire/T:.1f}%)")
    print(f"  Transitions w/ change  : {frames_that_changed} / {T-1}  ({100*frames_that_changed/(T-1):.1f}%)")
    print(f"  Unique fire-count vals : {unique_totals}")
    print(f"  Mean temporal variance : {temporal_var:.10f}")
    print(f"  All frames IDENTICAL?  : {all_same}  <-- TRUE means static copy bug!")

    if all_same:
        print("  *** CRITICAL: MODIS_FIRE_T1 is a static copy — same map at every timestep! ***")
    elif frames_that_changed == 0:
        print("  *** CRITICAL: No transitions detected — fire map never changes! ***")
    elif frames_with_fire < 10:
        print("  *** WARNING: Very few frames have any fire at all. ***")
    else:
        print("  OK: Fire map varies dynamically over time.")
