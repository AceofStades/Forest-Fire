import xarray as xr
import rioxarray  # Import to enable xarray to read GeoTIFFs easily
import numpy as np
import os

# --- 1. Define File Paths (Using paths provided by user) ---
ERA5_NC_PATH = "dataset/ERA5-Land/era5_resampled_1km.nc"
DEM_PATH = "dataset/resampled/dem_resampled.tif"
LULC_PATH = "dataset/resampled/lulc_resampled.tif"
GHS_PATH = "dataset/resampled/ghs_resampled.tif"
MODIS_PATH = "dataset/resampled/modis_raster.tif"
OUTPUT_NC_PATH = "dataset/final_feature_stack.nc"


# --- 2. Load Static GeoTIFFs and Create Static Dataset (FIXED) ---
def create_static_dataset():
    # FIX: Use [0] indexing to select the first band by position,
    # then use the simple .squeeze() to remove redundant dimensions safely.

    dem_da = rioxarray.open_rasterio(DEM_PATH)[0].squeeze(drop=True)
    lulc_da = rioxarray.open_rasterio(LULC_PATH)[0].squeeze(drop=True)
    ghs_da = rioxarray.open_rasterio(GHS_PATH)[0].squeeze(drop=True)
    modis_da = rioxarray.open_rasterio(MODIS_PATH)[0].squeeze(drop=True)

    # Rename variables and coordinate dimensions for cleaner stacking
    dem_da = dem_da.rename("DEM")
    lulc_da = lulc_da.rename("LULC")
    ghs_da = ghs_da.rename("GHS_BUILT")
    modis_da = modis_da.rename("MODIS_FIRE_T1")  # T1 will be used as a feature

    # Merge static data into a single Dataset
    static_ds = xr.Dataset(
        {"DEM": dem_da, "LULC": lulc_da, "GHS_BUILT": ghs_da, "MODIS_FIRE_T1": modis_da}
    )
    # No need for reset_coords, as the dimensions are now clean (latitude, longitude)
    return static_ds


# --- 3. Final Fusion and Save ---
def fuse_and_save_stack(static_ds, era5_nc_path, output_path):
    print("Loading ERA5 time-series data...")
    era5_ds = xr.open_dataset(era5_nc_path)

    # Final Merge: Xarray broadcasts static data across all ERA5 time steps.
    print("Merging static features with time-series data...")
    # The merge aligns automatically on shared latitude/longitude dimensions
    final_fused_dataset = era5_ds.merge(static_ds)

    # Create encoding dictionary for compression
    encoding = {
        var: {"zlib": True, "complevel": 5} for var in final_fused_dataset.data_vars
    }

    print(f"Saving final feature stack to {output_path}")
    final_fused_dataset.to_netcdf(output_path, format="NETCDF4", encoding=encoding)
    print("Fusion complete. Final feature stack is ready.")


# --- Execute ---
static_ds = create_static_dataset()
fuse_and_save_stack(static_ds, ERA5_NC_PATH, OUTPUT_NC_PATH)
