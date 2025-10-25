import xarray as xr
import rioxarray  # Ensures GeoTIFFs can be opened by xarray
import numpy as np
import os
import dask.diagnostics
from tqdm import tqdm  # Assuming tqdm is installed

# --- 1. Define File Paths ---
ERA5_NC_PATH = "dataset/ERA5-Land/era5_resampled_1km.nc"
DEM_PATH = "dataset/resampled/dem_resampled.tif"
LULC_PATH = "dataset/resampled/lulc_resampled.tif"
GHS_PATH = "dataset/resampled/ghs_resampled.tif"
MODIS_PATH = "dataset/resampled/modis_raster.tif"
OUTPUT_NC_PATH = "dataset/final_feature_stack2.nc"  # Using original output name again


# --- 2. Load Static GeoTIFFs and Prepare for Alignment ---
def create_static_dataset(era5_template_ds):
    print("Loading static GeoTIFF datasets...")

    def load_static_data(path, name, interp_method):
        # Open using rioxarray, select first band, remove band dim
        da = rioxarray.open_rasterio(path, masked=True).sel(band=1, drop=True)
        # Rename spatial coords to match ERA5
        da = da.rename({"x": "longitude", "y": "latitude"})
        da.name = name
        # Drop spatial_ref
        if "spatial_ref" in da.coords:
            da = da.drop_vars("spatial_ref")

        # *** CRITICAL ALIGNMENT STEP using interp_like ***
        print(f"  Aligning {name} using '{interp_method}' interpolation...")
        # Use the provided interpolation method (linear or nearest)
        da_aligned = da.interp_like(era5_template_ds, method=interp_method)

        # Ensure data type consistency after interpolation
        if name in ["LULC", "MODIS_FIRE_T1"]:
            # Fill potential NaNs from interpolation (especially edges) with 0
            da_aligned = da_aligned.fillna(0).astype(np.int16)
        else:
            # Fill potential NaNs for continuous data (DEM, GHS) with 0
            da_aligned = da_aligned.fillna(0).astype(np.float32)

        return da_aligned

    # Load each static layer, passing the ERA5 dataset as template
    # *** CORRECTED METHOD: Use 'linear' for continuous data ***
    dem_da = load_static_data(DEM_PATH, "DEM", "linear")
    lulc_da = load_static_data(LULC_PATH, "LULC", "nearest")
    ghs_da = load_static_data(GHS_PATH, "GHS_BUILT", "linear")
    modis_da = load_static_data(MODIS_PATH, "MODIS_FIRE_T1", "nearest")

    # Combine the *already aligned* static DataArrays
    static_ds = xr.merge([dem_da, lulc_da, ghs_da, modis_da])
    print("Static datasets loaded and aligned.")
    return static_ds


# --- 3. Final Fusion and Save ---
def fuse_and_save_stack(output_path):
    print("Loading ERA5 time-series data...")
    # Load ERA5 with chunks
    era5_ds = xr.open_dataset(ERA5_NC_PATH, chunks={"valid_time": 50})

    # Drop spatial_ref from ERA5 if it exists
    if "spatial_ref" in era5_ds.coords:
        era5_ds = era5_ds.drop_vars("spatial_ref")

    # Create the static dataset, aligning it to the ERA5 grid during creation
    static_ds_aligned = create_static_dataset(era5_ds)  # Pass era5_ds as template

    print("Merging aligned static features with time-series data...")
    # Perform the final merge
    final_fused_dataset = xr.merge([era5_ds, static_ds_aligned])

    # Create encoding dictionary for compression
    encoding = {
        var: {"zlib": True, "complevel": 5} for var in final_fused_dataset.data_vars
    }

    # Ensure output directory exists and handle old file
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Saving final feature stack to {output_path}...")

    # Use Dask progress bar for saving
    with dask.diagnostics.ProgressBar():
        final_fused_dataset.to_netcdf(output_path, format="NETCDF4", encoding=encoding)

    print("Fusion complete. Final feature stack is ready.")
    print("\n--- Final Dataset Structure ---")
    print(final_fused_dataset)  # Print structure to verify


# --- Execute ---
fuse_and_save_stack(OUTPUT_NC_PATH)
