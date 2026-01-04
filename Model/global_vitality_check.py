import numpy as np
import xarray as xr

# --- Configuration ---
# Path lil dataset l-master
FILE_PATH = "dataset/final_feature_stack_MASTER.nc"


def check_global_vitality():
    # Binbdaw nshoufo l-data kamla
    print(f"--- Global Vitality Check for: {FILE_PATH} ---")

    try:
        ds = xr.open_dataset(FILE_PATH)

        for var in ds.data_vars:
            data = ds[var].values
            total_pixels = data.size

            # N-countiw sh7al mn pixel fih l-data (mashi zero)
            non_zero_count = np.count_nonzero(data)
            non_zero_pct = (non_zero_count / total_pixels) * 100

            # Nshoufo l-min o l-max d kolshi
            v_min = np.nanmin(data)
            v_max = np.nanmax(data)

            print(f"\n[Variable: {var}]")
            print(
                f"  - Status: {'L-khir fih!' if non_zero_count > 0 else 'Hada khawi (All Zeros)!!!'}"
            )
            print(f"  - Non-Zero Pixels: {non_zero_count:,} ({non_zero_pct:.4f}%)")
            print(f"  - Range: [{v_min:.2f} to {v_max:.2f}]")

            # Logic dyal MODIS (Fire)
            if var == "MODIS_FIRE_T1":
                fire_days = np.sum(data.max(axis=(1, 2)) > 0)
                print(f"  - Days with Fire: {fire_days} / {ds.valid_time.size}")

        print("\n--- Check Salat ---")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_global_vitality()
