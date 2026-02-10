import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

NC_PATH = "dataset/final_feature_stack_MASTER.nc"


def inspect_raw_data():
    print(f"Opening {NC_PATH}...")
    ds = xr.open_dataset(NC_PATH, engine="h5netcdf")

    fire_var = "MODIS_FIRE_T1"
    if fire_var not in ds.data_vars:
        print(f"Error: {fire_var} not found in dataset.")
        return

    print("Loading Fire Data into memory...")
    fire_data = ds[fire_var].values  # Shape: (Time, Y, X)

    T, H, W = fire_data.shape
    print(f"Data Shape: Time={T}, H={H}, W={W}")

    # 1. Check for ANY change over time
    print("\n--- Test 1: Did the fire move at all? ---")

    # Compare frame 0 vs frame 10, 0 vs 100, etc.
    changes = []
    for t in [1, 8, 24, 100]:
        if t < T:
            diff = np.abs(fire_data[0] - fire_data[t]).sum()
            changes.append(diff)
            print(f"Difference between T=0 and T={t}: {diff} pixels changed")

    if sum(changes) == 0:
        print("\nCRITICAL FAILURE: The fire mask is IDENTICAL across all time steps.")
        print("Your NetCDF file contains a static image repeated 1400 times.")
        print("The model is correct (1.0 IoU) because the reality is frozen.")
        return

    # 2. Visualize a pixel timeline
    print("\n--- Test 2: Pixel History ---")
    # Find a pixel that burns
    flat_data = fire_data.reshape(T, -1)
    # Find pixel with max variance (most activity)
    active_pixel_idx = np.argmax(np.var(flat_data, axis=0))

    # Convert back to (y, x)
    y_idx, x_idx = np.unravel_index(active_pixel_idx, (H, W))

    pixel_timeline = fire_data[:, y_idx, x_idx]

    print(f"Inspecting active pixel at ({y_idx}, {x_idx})")
    print(f"Values first 50 steps: {pixel_timeline[:50]}")

    plt.figure(figsize=(10, 4))
    plt.plot(pixel_timeline)
    plt.title(f"Timeline of Pixel ({y_idx}, {x_idx})")
    plt.xlabel("Time Step")
    plt.ylabel("Fire Value")
    plt.savefig("pixel_timeline_check.png")
    print("Saved timeline plot to 'pixel_timeline_check.png'")

    # 3. Check for exact duplicates in sequence
    print("\n--- Test 3: Lag Check ---")
    # Check if T and T+8 are identical
    diff_8 = np.abs(fire_data[:-8] - fire_data[8:]).sum()
    print(f"Total pixel differences with 8-step lag (24h): {diff_8}")

    if diff_8 == 0:
        print("VERDICT: Data is static over 24h windows.")
    else:
        print(
            "VERDICT: Data is dynamic. The persistence check script might have had a bug."
        )


if __name__ == "__main__":
    inspect_raw_data()
