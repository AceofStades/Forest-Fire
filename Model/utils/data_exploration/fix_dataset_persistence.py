import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm

input_path = "dataset/final_feature_stack_DYNAMIC_new.nc"
output_path = "dataset/final_feature_stack_DYNAMIC_interpolated.nc"

print(f"Loading dataset from {input_path}...")
ds = xr.open_dataset(input_path, engine="h5netcdf")
ds = ds.load()

fire = ds["MODIS_FIRE_T1"].values

total_steps = fire.shape[0]
persistence_hours = 24  # Let a fire "burn" for up to 24 hours after a satellite detection

print(f"Applying persistence (forward-filling fire for {persistence_hours} hours)...")

# We iterate through time. If a pixel is on fire at time T, 
# we set it to on fire for T+1, T+2, ... T+persistence_hours, UNLESS it is already on fire there.
# To do this efficiently, we can use a rolling max or an explicit loop.

# Using a loop to be explicit and memory safe
new_fire = np.copy(fire)

# Find all indices where fire is 1
times, lats, lons = np.where(fire > 0)

for t, lat, lon in tqdm(zip(times, lats, lons), total=len(times), desc="Interpolating"):
    # Forward fill for 'persistence_hours'
    end_t = min(t + persistence_hours, total_steps)
    new_fire[t:end_t, lat, lon] = 1.0

# Verify changes
print("\n--- Stats Before ---")
print("Total fire pixels:", fire.sum())
print("Frames with fire:", (fire.sum(axis=(1,2)) > 0).sum(), "/", total_steps)

print("\n--- Stats After ---")
print("Total fire pixels:", new_fire.sum())
print("Frames with fire:", (new_fire.sum(axis=(1,2)) > 0).sum(), "/", total_steps)

# Assign back to dataset
ds["MODIS_FIRE_T1"].values = new_fire

# Save to new file
print(f"\nSaving to {output_path}...")
ds.to_netcdf(output_path, engine="h5netcdf")
print("Done!")

