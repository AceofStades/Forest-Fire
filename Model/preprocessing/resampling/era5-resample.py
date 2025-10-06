import pandas as pd
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os

# --- 1. Define the Corrected Target Grid (1 km resolution) ---
target_crs = "EPSG:4326"
target_resolution = 0.009  # Approximately 1 km in decimal degrees
target_bounds = rasterio.coords.BoundingBox(
    left=77.5, bottom=28.7, right=81.1, top=31.5
)
target_width = int(
    round((target_bounds.right - target_bounds.left) / target_resolution)
)
target_height = int(
    round((target_bounds.top - target_bounds.bottom) / target_resolution)
)

print(f"New Target Grid Dimensions: {target_width} x {target_height} pixels")
print(f"New Target Resolution: {target_resolution} degrees (approx 1 km)\n")


# --- 2. Resampling Function (Generates Resampled DataArrays) ---
def resample_era5_to_1km(input_nc_path):
    print("Starting 1km resampling...")
    ds = xr.open_dataset(input_nc_path)
    num_timesteps = ds.dims["valid_time"]

    # List to hold the resampled data for all time steps
    resampled_datasets = []

    for i in range(num_timesteps):
        # Process a single time step
        print(f"Processing step {i + 1}/{num_timesteps}: {ds['valid_time'].values[i]}")

        ds_chunk = ds.isel(valid_time=i).drop_vars("valid_time")
        variables_to_stack = list(ds_chunk.data_vars)
        stacked_data = np.stack(
            [ds_chunk[var].values for var in variables_to_stack], axis=0
        )

        # Get the original ERA5 transform
        x_res = abs(
            ds_chunk.coords["longitude"].values[1]
            - ds_chunk.coords["longitude"].values[0]
        )
        y_res = abs(
            ds_chunk.coords["latitude"].values[1]
            - ds_chunk.coords["latitude"].values[0]
        )
        era5_transform = rasterio.transform.from_origin(
            ds_chunk.coords["longitude"].values[0],
            ds_chunk.coords["latitude"].values[0],
            x_res,
            y_res,
        )

        # Prepare the output array
        output_data = np.empty(
            (len(variables_to_stack), target_height, target_width), dtype=np.float32
        )

        # Reproject (Resample) to the 1 km target grid
        reproject(
            source=stacked_data,
            destination=output_data,
            src_transform=era5_transform,
            src_crs="EPSG:4326",
            dst_transform=rasterio.transform.from_bounds(
                target_bounds.left,
                target_bounds.bottom,
                target_bounds.right,
                target_bounds.top,
                target_width,
                target_height,
            ),
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )

        # Create a new xarray DataArray for the resampled data
        resampled_ds = xr.Dataset(
            {
                var: (("latitude", "longitude"), output_data[j])
                for j, var in enumerate(variables_to_stack)
            },
            coords={
                "valid_time": ds["valid_time"].values[i],
                "latitude": np.linspace(
                    target_bounds.top, target_bounds.bottom, target_height
                ),
                "longitude": np.linspace(
                    target_bounds.left, target_bounds.right, target_width
                ),
            },
        )
        resampled_datasets.append(resampled_ds)

    # Concatenate all time steps back into one Xarray Dataset
    final_ds = xr.concat(resampled_datasets, dim="valid_time")
    return final_ds


# --- 3. Run and Save ---
era5_nc_path = "dataset/ERA5-Land/final-era5_rechunked.nc"
output_nc_path = "dataset/ERA5-Land/era5_resampled_1km.nc"

final_era5_ds = resample_era5_to_1km(era5_nc_path)

# Save the final dataset with compression
encoding = {var: {"zlib": True, "complevel": 5} for var in final_era5_ds.data_vars}
final_era5_ds.to_netcdf(output_nc_path, format="NETCDF4", encoding=encoding)

print(f"\nFinal 1km resampled ERA5 data saved to: {output_nc_path}")
