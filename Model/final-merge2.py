import os

import dask.diagnostics
import numpy as np
import rioxarray  # ensures geotiffs can be opened by xarray
import xarray as xr
from tqdm import tqdm  # assuming tqdm is installed

# --- 1. define file paths ---
era5_nc_path = "dataset/era5-land/era5_resampled_1km.nc"
dem_path = "dataset/resampled-fix/dem_resampled.tif"
lulc_path = "dataset/resampled-fix/lulc_resampled.tif"
ghs_path = "dataset/resampled-fix/ghs_resampled.tif"
modis_path = "dataset/resampled/modis_raster.tif"
output_nc_path = "dataset/final_feature_stack2.nc"  # using original output name again


# --- 2. load static geotiffs and prepare for alignment ---
def create_static_dataset(era5_template_ds):
    print("loading static geotiff datasets...")

    def load_static_data(path, name, interp_method):
        # open using rioxarray, select first band, remove band dim
        da = rioxarray.open_rasterio(path, masked=true).sel(band=1, drop=true)
        # rename spatial coords to match era5
        da = da.rename({"x": "longitude", "y": "latitude"})
        da.name = name
        # drop spatial_ref
        if "spatial_ref" in da.coords:
            da = da.drop_vars("spatial_ref")

        # *** critical alignment step using interp_like ***
        print(f"  aligning {name} using '{interp_method}' interpolation...")
        # use the provided interpolation method (linear or nearest)
        da_aligned = da.interp_like(era5_template_ds, method=interp_method)

        # ensure data type consistency after interpolation
        if name in ["lulc", "modis_fire_t1"]:
            # fill potential nans from interpolation (especially edges) with 0
            da_aligned = da_aligned.fillna(0).astype(np.int16)
        else:
            # fill potential nans for continuous data (dem, ghs) with 0
            da_aligned = da_aligned.fillna(0).astype(np.float32)

        return da_aligned

    # load each static layer, passing the era5 dataset as template
    # *** corrected method: use 'linear' for continuous data ***
    dem_da = load_static_data(dem_path, "dem", "linear")
    lulc_da = load_static_data(lulc_path, "lulc", "nearest")
    ghs_da = load_static_data(ghs_path, "ghs_built", "linear")
    modis_da = load_static_data(modis_path, "modis_fire_t1", "nearest")

    # combine the *already aligned* static dataarrays
    static_ds = xr.merge([dem_da, lulc_da, ghs_da, modis_da])
    print("static datasets loaded and aligned.")
    return static_ds


# --- 3. final fusion and save ---
def fuse_and_save_stack(output_path):
    print("loading era5 time-series data...")
    # load era5 with chunks
    era5_ds = xr.open_dataset(era5_nc_path, chunks={"valid_time": 50})

    # drop spatial_ref from era5 if it exists
    if "spatial_ref" in era5_ds.coords:
        era5_ds = era5_ds.drop_vars("spatial_ref")

    # create the static dataset, aligning it to the era5 grid during creation
    static_ds_aligned = create_static_dataset(era5_ds)  # pass era5_ds as template

    print("merging aligned static features with time-series data...")
    # perform the final merge
    final_fused_dataset = xr.merge([era5_ds, static_ds_aligned])

    # create encoding dictionary for compression
    encoding = {
        var: {"zlib": true, "complevel": 5} for var in final_fused_dataset.data_vars
    }

    # ensure output directory exists and handle old file
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"saving final feature stack to {output_path}...")

    # use dask progress bar for saving
    with dask.diagnostics.ProgressBar():
        final_fused_dataset.to_netcdf(output_path, format="netcdf4", encoding=encoding)

    print("fusion complete. final feature stack is ready.")
    print("\n--- final dataset structure ---")
    print(final_fused_dataset)  # print structure to verify


# --- execute ---
fuse_and_save_stack(output_nc_path)
