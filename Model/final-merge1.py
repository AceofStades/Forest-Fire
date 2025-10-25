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
OUTPUT_NC_PATH = "dataset/final_feature_stack1.nc"  # Using new output name


# --- 2. Load Static GeoTIFFs and Force Alignment (CORRECTED) ---
def create_static_dataset_aligned(target_lat_coords, target_lon_coords):
    print("Loading static GeoTIFF datasets and aligning coordinates...")

    def load_and_force_align(path, name, target_lat, target_lon):
        # Open using rioxarray, select first band, remove band dim
        da = rioxarray.open_rasterio(path, masked=True).sel(band=1, drop=True)

        # *** CRITICAL FIX: Rename dimensions BEFORE assigning coords ***
        # Rename 'y' to 'latitude' and 'x' to 'longitude' to match ERA5
        da = da.rename({"y": "latitude", "x": "longitude"})

        # Now assign the EXACT coordinates from the ERA5 dataset
        # This works because the dimension names now match the coordinate names
        da = da.assign_coords(latitude=target_lat, longitude=target_lon)

        # Assign a name to the DataArray
        da.name = name

        # Remove spatial_ref coord if it exists
        if "spatial_ref" in da.coords:
            da = da.drop_vars("spatial_ref")

        # Ensure data type consistency
        if name in ["LULC", "MODIS_FIRE_T1"]:
            da = da.astype(np.int16)
        else:
            da = da.astype(np.float32)

        return da

    # Load each static layer and force its coordinates
    dem_da = load_and_force_align(DEM_PATH, "DEM", target_lat_coords, target_lon_coords)
    lulc_da = load_and_force_align(
        LULC_PATH, "LULC", target_lat_coords, target_lon_coords
    )
    ghs_da = load_and_force_align(
        GHS_PATH, "GHS_BUILT", target_lat_coords, target_lon_coords
    )
    modis_da = load_and_force_align(
        MODIS_PATH, "MODIS_FIRE_T1", target_lat_coords, target_lon_coords
    )

    # Combine into a single static Dataset
    static_ds = xr.merge([dem_da, lulc_da, ghs_da, modis_da])
    print("Static datasets loaded and aligned.")
    return static_ds


# --- 3. Final Fusion and Save ---
def fuse_and_save_stack(output_path):
    print("Loading ERA5 time-series data...")
    # Load ERA5 with chunks
    era5_ds = xr.open_dataset(ERA5_NC_PATH, chunks={"valid_time": 50})

    # Extract target coordinates from ERA5
    target_lat_coords = era5_ds["latitude"]
    target_lon_coords = era5_ds["longitude"]

    # Drop spatial_ref from ERA5 if it exists
    if "spatial_ref" in era5_ds.coords:
        era5_ds = era5_ds.drop_vars("spatial_ref")

    # Create the static dataset with forced coordinate alignment
    static_ds_aligned = create_static_dataset_aligned(
        target_lat_coords, target_lon_coords
    )

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
