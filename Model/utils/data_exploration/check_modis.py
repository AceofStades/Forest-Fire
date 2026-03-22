import pandas as pd
import xarray as xr
import numpy as np

print("--- Checking final-modis.csv ---")
df = pd.read_csv("dataset/MODIS/final-modis.csv")
print("Columns:", df.columns.tolist())
if "acq_date" in df.columns:
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    print(f"Date range: {df['acq_date'].min()} to {df['acq_date'].max()}")
    print("Unique dates with fire:", df["acq_date"].nunique())
    print("Total fire occurrences:", len(df))
    print("Counts per month:")
    print(df["acq_date"].dt.month.value_counts().sort_index())
    
print("\n--- Checking MODIS_FIRE_T1 in final_feature_stack_DYNAMIC_new.nc ---")
ds = xr.open_dataset("dataset/final_feature_stack_DYNAMIC_new.nc", engine="h5netcdf")
print("Time dimension shape:", ds.sizes["valid_time"])
print("Time range:", ds["valid_time"].min().values, "to", ds["valid_time"].max().values)

fire = ds["MODIS_FIRE_T1"]
fire_per_step = fire.sum(dim=["latitude", "longitude"]).values
steps_with_fire = (fire_per_step > 0).sum()
print("Steps with fire:", steps_with_fire)
print("Total fire pixels:", fire_per_step.sum())

# Check mapping: how many fires on corresponding dates?
times = pd.to_datetime(ds["valid_time"].values)
fire_ts = pd.Series(fire_per_step, index=times)
print("\nFires per month in .nc dataset:")
print(fire_ts[fire_ts > 0].groupby(fire_ts[fire_ts > 0].index.month).sum())

