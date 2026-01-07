import numpy as np
import xarray as xr

# FILE_PATH = "dataset/final_feature_stack.nc"
# FILE_PATH = "dataset/final_feature_stack1.nc"
FILE_PATH = "../dataset/final_feature_stack_MASTER.nc"


def validate_dataset(path):
    print(f"--- Starting Validation for: {path} ---\n")
    ds = xr.open_dataset(path)

    # 1. Dimension & Coordinate Check
    print(f"[1] Dimensions: {dict(ds.dims)}")
    print(f"[2] Variables: {list(ds.data_vars)}")

    # 2. Global NaN Check
    print("\n[3] Checking for Missing Values (NaNs)...")
    total_nans = 0
    for var in ds.data_vars:
        nan_count = ds[var].isnull().sum().values
        total_nans += nan_count
        if nan_count > 0:
            print(f"    - WARNING: {var} contains {nan_count} NaNs!")
        else:
            print(f"    - {var}: Clean (0 NaNs)")

    if total_nans == 0:
        print("    >>> Global Status: All pixels have valid numerical data.")

    # 3. Value Range & Data Integrity Check
    print("\n[4] Data Range Statistics:")
    for var in ds.data_vars:
        v_min = ds[var].min().values
        v_max = ds[var].max().values
        v_mean = ds[var].mean().values
        print(
            f"    - {var:15} | Min: {v_min:8.2f} | Max: {v_max:8.2f} | Mean: {v_mean:8.2f}"
        )

    # 4. Integrity of Binary/Categorical Layers
    print("\n[5] Logic Checks:")

    # MODIS check: should only be 0 or 1
    unique_modis = np.unique(ds["MODIS_FIRE_T1"].values)
    print(f"    - MODIS_FIRE_T1 unique values: {unique_modis}")
    if not np.all(np.isin(unique_modis, [0, 1])):
        print("      CRITICAL WARNING: MODIS contains values other than 0 and 1!")

    # LULC check: check if it's within expected range (0-255 usually)
    if ds["LULC"].max() > 255:
        print("    - WARNING: LULC has values > 255. Verify classification scale.")

    # ERA5 Temperature check (K to C check)
    if "t2m" in ds.data_vars:
        if ds["t2m"].mean() > 200:
            print("    - Note: Temperature (t2m) is in Kelvin.")
        else:
            print("    - Note: Temperature (t2m) is in Celsius.")

    # 5. Temporal Continuity
    time_diffs = np.diff(ds.valid_time.values) / np.timedelta64(1, "h")
    if not np.all(time_diffs == 1.0):
        print("\n[6] WARNING: Time steps are not perfectly hourly! Check ERA5 source.")
    else:
        print("\n[6] Time continuity: Perfect (Hourly).")

    print("\n--- Validation Finished ---")


if __name__ == "__main__":
    validate_dataset(FILE_PATH)
