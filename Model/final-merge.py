import xarray as xr
import rioxarray
import numpy as np
import os
import shutil
import dask.diagnostics
from tqdm import tqdm

# --- 1. Define File Paths ---
ERA5_NC_PATH = "dataset/ERA5-Land/era5_resampled_1km.nc"
DEM_PATH = "dataset/resampled/dem_resampled.tif"
LULC_PATH = "dataset/resampled/lulc_resampled.tif"
GHS_PATH = "dataset/resampled/ghs_resampled.tif"
MODIS_PATH = "dataset/resampled/modis_raster.tif"
OUTPUT_NC_PATH = "dataset/final_feature_stack.nc"


# --- 2. Load Static GeoTIFFs and Create Static Dataset ---
def create_static_dataset(era5_time_coords, era5_lon_coords, era5_lat_coords):
    print("Broadcasting static features across time dimension...")

    def load_and_align_static(path, new_name):
        # 1. Load GeoTIFF and prepare dimensions
        da = rioxarray.open_rasterio(path)[0].squeeze(drop=True)
        da = da.rename({"x": "longitude", "y": "latitude"}).rename(new_name)

        # 2. Reindex (initial alignment) - THIS IS THE PROBLEM STEP
        # We round to 10 decimal places to eliminate floating-point errors
        da["longitude"] = np.round(da["longitude"].values, 10)
        da["latitude"] = np.round(da["latitude"].values, 10)

        # 3. Reindex using Nearest Neighbor - This is the step that failed for LULC/GHS
        da_aligned = da.reindex(
            latitude=era5_lat_coords,
            longitude=era5_lon_coords,
            method="nearest",  # Method is too rigid, causing gaps
        )

        # 4. Expand the static 2D array across the full time dimension
        return da_aligned.expand_dims(valid_time=era5_time_coords)

    # Load ERA5 dataset temporarily to get the exact coordinate values
    with xr.open_dataset(ERA5_NC_PATH) as era5_ds_temp:
        era5_time_coords = era5_ds_temp["valid_time"]
        # Round the target ERA5 coordinates as well for consistency
        era5_lon_coords = np.round(era5_ds_temp["longitude"].values, 10)
        era5_lat_coords = np.round(era5_ds_temp["latitude"].values, 10)

    # Load all static data, forcing them to have the 'valid_time' dimension
    static_ds = xr.Dataset(
        {
            "DEM": load_and_align_static(DEM_PATH, "DEM"),
            "LULC": load_and_align_static(LULC_PATH, "LULC"),
            "GHS_BUILT": load_and_align_static(GHS_PATH, "GHS_BUILT"),
            "MODIS_FIRE_T1": load_and_align_static(MODIS_PATH, "MODIS_FIRE_T1"),
        }
    )
    return static_ds.drop_vars("band", errors="ignore")


# --- 3. Final Fusion and Save (SEQUENTIAL FIX with Progress Bar) ---
def fuse_and_save_stack_safe(output_path):
    print("Loading ERA5 time-series data...")
    # Load with Dask chunks for lazy loading and memory safety
    era5_ds = xr.open_dataset(ERA5_NC_PATH).chunk({"valid_time": 50})

    # Get coordinates and align the static data
    static_ds = create_static_dataset(
        era5_ds["valid_time"], era5_ds["longitude"], era5_ds["latitude"]
    )

    # Filter out the 'band' variable if it somehow persisted
    era5_ds = era5_ds.drop_vars("band", errors="ignore")

    print("Merging static features with time-series data...")
    final_fused_dataset = era5_ds.merge(static_ds)

    # Filter out the redundant 'band' coordinate/variable from the final dataset
    final_fused_dataset = final_fused_dataset.drop_vars("band", errors="ignore")

    # Create encoding dictionary for compression
    encoding = {
        var: {"zlib": True, "complevel": 5} for var in final_fused_dataset.data_vars
    }

    # Ensure the output directory exists and remove old file
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Saving final feature stack to {output_path} (sequential write)...")

    with dask.diagnostics.ProgressBar():
        # Execute the computation and save the data sequentially
        final_fused_dataset.to_netcdf(output_path, format="NETCDF4", encoding=encoding)

    print("Fusion complete. Final feature stack is ready.")


# --- Execute ---
# NOTE: The outer script will call this function.
fuse_and_save_stack_safe(OUTPUT_NC_PATH)
