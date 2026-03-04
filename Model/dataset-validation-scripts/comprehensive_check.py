import os

import numpy as np
import xarray as xr

DATASETS = {
    "MASTER": "dataset/final_feature_stack_MASTER.nc",
    "DYNAMIC": "dataset/final_feature_stack_DYNAMIC.nc",
}


def check_dataset(name, path):
    print(f"\n{'=' * 40}")
    print(f"Checking {name} Dataset: {path}")
    print(f"{'=' * 40}")

    if not os.path.exists(path):
        print("❌ File not found.")
        return

    try:
        ds = xr.open_dataset(path, engine="h5netcdf")

        # 1. Check Dimensions
        print(f"Dimensions: {dict(ds.dims)}")

        # 2. Check Fire Variable (MODIS_FIRE_T1)
        if "MODIS_FIRE_T1" not in ds.data_vars:
            print("❌ MODIS_FIRE_T1 variable missing!")
            return

        fire = ds["MODIS_FIRE_T1"].values
        total_pixels = fire.size
        fire_pixels = np.sum(fire > 0)

        print(f"\n[Fire Statistics]")
        print(f"Total Pixels: {total_pixels:,}")
        print(
            f"Fire Pixels:  {fire_pixels:,} ({fire_pixels / total_pixels * 100:.4f}%)"
        )

        if fire_pixels == 0:
            print("❌ CRITICAL: No fire pixels found in the entire dataset!")
            return

        # 3. Check for Static Fire Data (Frozen Time)
        print("\n[Dynamics Check]")
        # Compare first frame with 10th, 50th, etc.
        t_steps = ds.dims["valid_time"]
        if t_steps > 1:
            diff_1 = np.abs(fire[0] - fire[1]).sum()
            diff_10 = np.abs(fire[0] - fire[min(10, t_steps - 1)]).sum()

            print(f"Pixel changes (T=0 vs T=1):  {diff_1}")
            print(f"Pixel changes (T=0 vs T=10): {diff_10}")

            if diff_10 == 0:
                print("❌ CRITICAL: Fire data appears STATIC (no changes over time).")
            else:
                print("✅ Fire data is DYNAMIC (changes detected).")

        # 4. Check Weather/Dynamic Features
        # Pick a random feature like 't2m' (Temperature) or 'u10' (Wind)
        check_var = "t2m" if "t2m" in ds.data_vars else list(ds.data_vars.keys())[0]
        if check_var != "MODIS_FIRE_T1":
            data_var = ds[check_var].values
            var_diff = np.abs(data_var[0] - data_var[min(10, t_steps - 1)]).sum()
            print(f"\n[Feature Check: {check_var}]")
            print(f"Changes (T=0 vs T=10): {var_diff}")
            if var_diff == 0:
                print(f"⚠️ Warning: Feature {check_var} appears static.")
            else:
                print(f"✅ Feature {check_var} is dynamic.")

    except Exception as e:
        print(f"❌ Error reading dataset: {e}")


if __name__ == "__main__":
    for name, path in DATASETS.items():
        check_dataset(name, path)
