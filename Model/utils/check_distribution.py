import numpy as np
import xarray as xr

NC_PATH = "dataset/final_feature_stack_DYNAMIC.nc"


def check_distribution():
    print(f"Loading {NC_PATH}...")
    ds = xr.open_dataset(NC_PATH, engine="h5netcdf")
    fire_data = ds["MODIS_FIRE_T1"].values  # (Time, Lat, Lon)

    _, H, W = fire_data.shape
    split_row = int(H * 0.80)

    # Split
    north_fire = fire_data[:, :split_row, :]
    south_fire = fire_data[:, split_row:, :]

    print("\n--- FIRE DISTRIBUTION ---")
    print(f"Total Pixels in North (Train): {north_fire.sum()}")
    print(f"Total Pixels in South (Val):   {south_fire.sum()}")

    if south_fire.sum() == 0:
        print("\nCRITICAL PROBLEM: The South region (Validation) has NO fires.")
        print(
            "Your model cannot be validated spatially because there is nothing to predict."
        )
        print("RECOMMENDATION: Switch to a Temporal Split or Random Spatial Split.")
    elif south_fire.sum() < 100:
        print("\nWARNING: Very few fires in South region. Validation will be unstable.")
    else:
        print("\nDistribution looks okay. Model might just be learning slowly.")


if __name__ == "__main__":
    check_distribution()
