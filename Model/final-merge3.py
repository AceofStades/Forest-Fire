import os

import dask.diagnostics
import numpy as np
import rioxarray  # ensures GeoTIFFs can be opened by xarray
import xarray as xr
from tqdm import tqdm

# --- 1. Define File Paths (Updated to resampled-fix) ---
ERA5_NC_PATH = "dataset/ERA5-Land/era5_resampled_1km.nc"
DEM_PATH = "dataset/resampled-fix/dem_resampled.tif"
LULC_PATH = "dataset/resampled-fix/lulc_resampled.tif"
GHS_PATH = "dataset/resampled-fix/ghs_resampled.tif"
MODIS_PATH = "dataset/resampled/modis_raster.tif"  # Keep your current MODIS path
OUTPUT_NC_PATH = "dataset/final_feature_stack3.nc"


# --- 2. Load and Align Function ---
def create_static_dataset(era5_template_ds):
    print("Loading static GeoTIFF datasets...")

    def load_static_data(path, name, interp_method):
        # Open using rioxarray
        da = rioxarray.open_rasterio(path, masked=True).sel(band=1, drop=True)

        # Rename spatial coords to match ERA5 naming convention
        da = da.rename({"x": "longitude", "y": "latitude"})
        da.name = name

        # Drop spatial_ref to avoid NetCDF conflicts
        if "spatial_ref" in da.coords:
            da = da.drop_vars("spatial_ref")

        print(f"  Aligning {name} using '{interp_method}' interpolation...")
        # Align grid to ERA5 template
        da_aligned = da.interp_like(era5_template_ds, method=interp_method)

        # Handle NaNs (from edges) and enforce data types
        if name in ["lulc", "modis_fire_t1"]:
            da_aligned = da_aligned.fillna(0).astype(np.int16)
        else:
            da_aligned = da_aligned.fillna(0).astype(np.float32)

        return da_aligned

    # Load each layer
    dem_da = load_static_data(DEM_PATH, "DEM", "linear")
    lulc_da = load_static_data(LULC_PATH, "LULC", "nearest")
    ghs_da = load_static_data(GHS_PATH, "GHS_BUILT", "linear")
    modis_da = load_static_data(MODIS_PATH, "MODIS_FIRE_T1", "nearest")

    # --- CRITICAL FIX: Broadcast MODIS to Time Dimension ---
    # xarray.broadcast expands the 2D MODIS data across the 3D ERA5 time dimension
    # This allows for time-shifting (t -> t+1) during training.
    print("  Broadcasting MODIS_FIRE_T1 across the time dimension...")
    modis_temporal, _ = xr.broadcast(modis_da, era5_template_ds.valid_time)
    modis_temporal.name = "MODIS_FIRE_T1"

    # Merge aligned static data and temporal MODIS
    static_ds = xr.merge([dem_da, lulc_da, ghs_da, modis_temporal])
    print("Static and Fire datasets loaded and aligned.")
    return static_ds


# --- 3. Final Fusion and Save ---
def fuse_and_save_stack(output_path):
    print("Loading ERA5 time-series data...")
    # Load ERA5 with dask chunks for memory efficiency
    era5_ds = xr.open_dataset(ERA5_NC_PATH, chunks={"valid_time": 50})

    if "spatial_ref" in era5_ds.coords:
        era5_ds = era5_ds.drop_vars("spatial_ref")

    # Create aligned static dataset (includes the new 3D MODIS layer)
    static_ds_aligned = create_static_dataset(era5_ds)

    print("Merging aligned features with time-series data...")
    final_fused_dataset = xr.merge([era5_ds, static_ds_aligned])

    # Encoding for NetCDF4 compression
    encoding = {
        var: {"zlib": True, "complevel": 5} for var in final_fused_dataset.data_vars
    }

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Saving final feature stack to {output_path}...")
    with dask.diagnostics.ProgressBar():
        # Using compute() if needed, but to_netcdf handles dask arrays automatically
        final_fused_dataset.to_netcdf(output_path, format="netcdf4", encoding=encoding)

    print("Fusion complete. Final feature stack is ready.")
    print("\n--- Final Dataset Structure Verification ---")
    # Verify that MODIS_FIRE_T1 now has (valid_time, latitude, longitude)
    print(final_fused_dataset.data_vars)


# --- Execute ---
if __name__ == "__main__":
    fuse_and_save_stack(OUTPUT_NC_PATH)
