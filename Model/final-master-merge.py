import os

import dask.diagnostics
import numpy as np
import rioxarray  # ensures GeoTIFFs can be opened by xarray
import xarray as xr
from tqdm import tqdm

# --- 1. Define File Paths ---
ERA5_NC_PATH = "dataset/ERA5-Land/era5_resampled_1km.nc"
DEM_PATH = "dataset/resampled-fix/dem_resampled.tif"
LULC_PATH = "dataset/resampled-fix/lulc_resampled.tif"
GHS_PATH = "dataset/resampled-fix/ghs_resampled.tif"
MODIS_PATH = "dataset/resampled/modis_raster.tif"
OUTPUT_NC_PATH = "dataset/final_feature_stack_MASTER.nc"


# --- 2. Master Alignment and Loading ---
def create_master_dataset(era5_template_ds):
    print("Loading and cleaning static GeoTIFF datasets...")

    def load_and_clean_static(path, name, interp_method):
        # Open using rioxarray
        da = rioxarray.open_rasterio(path, masked=True).sel(band=1, drop=True)
        da = da.rename({"x": "longitude", "y": "latitude"})
        da.name = name

        # FIX: Specific cleaning for GHS_BUILT where 255 is often "No Data"
        if name == "GHS_BUILT":
            da = da.where(da != 255, 0)

        # Drop spatial_ref to avoid NetCDF4 saving errors
        if "spatial_ref" in da.coords:
            da = da.drop_vars("spatial_ref")

        print(f"  Aligning {name} using '{interp_method}' interpolation...")
        # Initial alignment to the template grid
        da_aligned = da.interp_like(era5_template_ds, method=interp_method)

        # Fill standard NaNs with 0
        return da_aligned.fillna(0)

    # Load all static layers
    dem = load_and_clean_static(DEM_PATH, "DEM", "linear")
    lulc = load_and_clean_static(LULC_PATH, "LULC", "nearest")
    ghs = load_and_clean_static(GHS_PATH, "GHS_BUILT", "linear")
    modis = load_and_clean_static(MODIS_PATH, "MODIS_FIRE_T1", "nearest")

    # --- 3. THE MASTER CROP: Remove the artificial zero-padding ---
    # We use DEM to find where valid land data actually exists
    print("Calculating intersection mask to remove edge padding...")
    valid_mask = (dem > 0) & (dem.notnull())

    # Find coordinates where data is valid
    valid_coords = np.argwhere(valid_mask.values)
    lat_min, lat_max = valid_coords[:, 0].min(), valid_coords[:, 0].max()
    lon_min, lon_max = valid_coords[:, 1].min(), valid_coords[:, 1].max()

    print(
        f"  Cropping to interior bounds: Lat[{lat_min}:{lat_max}], Lon[{lon_min}:{lon_max}]"
    )

    # Slice the static layers
    dem = dem.isel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    lulc = lulc.isel(
        latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
    )
    ghs = ghs.isel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    modis = modis.isel(
        latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
    )

    # --- 4. BROADCASTING MODIS ---
    print("  Broadcasting MODIS_FIRE_T1 across the time dimension...")
    modis_temporal, _ = xr.broadcast(modis, era5_template_ds.valid_time)

    # Merge all static and temporal fire data
    static_ds = xr.merge(
        [
            dem.astype(np.float32),
            lulc.astype(np.int16),
            ghs.astype(np.float32),
            modis_temporal.astype(np.float32),
        ]
    )

    return static_ds, (lat_min, lat_max, lon_min, lon_max)


# --- 5. Final Fusion and Save ---
def fuse_master_stack(output_path):
    print("Loading ERA5 time-series data...")
    era5_ds = xr.open_dataset(ERA5_NC_PATH, chunks={"valid_time": 50})
    if "spatial_ref" in era5_ds.coords:
        era5_ds = era5_ds.drop_vars("spatial_ref")

    # Create the clean, cropped static dataset
    static_ds, slices = create_master_dataset(era5_ds)
    lat_min, lat_max, lon_min, lon_max = slices

    print("Merging features and applying master crop to weather data...")
    # Crop the weather data to match the static data intersection
    era5_cropped = era5_ds.isel(
        latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
    )

    # Perform final merge
    final_ds = xr.merge([era5_cropped, static_ds])

    # FORCE DIMENSION ORDER: (Time, Lat, Lon)
    # This is vital for PyTorch/Dataloader consistency
    final_ds = final_ds.transpose("valid_time", "latitude", "longitude")

    # Encoding for NetCDF4 compression
    encoding = {var: {"zlib": True, "complevel": 5} for var in final_ds.data_vars}

    # Handle directory and existing file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Saving Master Stack to {output_path}...")
    with dask.diagnostics.ProgressBar():
        final_ds.to_netcdf(output_path, format="netcdf4", encoding=encoding)

    print("\nFusion complete. The Master Stack is ready for training.")
    print("Final Grid Dimensions:", final_ds.dims)


if __name__ == "__main__":
    fuse_master_stack(OUTPUT_NC_PATH)
