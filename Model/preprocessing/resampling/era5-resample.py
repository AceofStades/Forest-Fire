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

# Coordinate labels must be the CENTRE of each cell, to match the raster the
# reproject below actually writes (from_bounds divides the extent into N cells
# of size extent/N). np.linspace(top, bottom, N) instead returns N points
# spanning the extent inclusive, i.e. spacing extent/(N-1): half a cell off at
# the corner and drifting to a full cell by the far edge.
_cell_lat = (target_bounds.top - target_bounds.bottom) / target_height
_cell_lon = (target_bounds.right - target_bounds.left) / target_width
target_lats = target_bounds.top - (np.arange(target_height) + 0.5) * _cell_lat
target_lons = target_bounds.left + (np.arange(target_width) + 0.5) * _cell_lon


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
        era5_lons = ds_chunk.coords["longitude"].values
        era5_lats = ds_chunk.coords["latitude"].values
        x_res = abs(era5_lons[1] - era5_lons[0])
        y_res = abs(era5_lats[1] - era5_lats[0])

        # from_origin() expects the OUTER CORNER of the top-left cell, but ERA5
        # coordinates are grid points, i.e. cell CENTRES. Passing the centre
        # directly places every value half a cell (0.05 deg ~ 5.5 km) south-east
        # of where it belongs, which is ~5 output pixels at the 1 km target grid.
        west_edge = era5_lons.min() - x_res / 2
        north_edge = era5_lats.max() + y_res / 2
        era5_transform = rasterio.transform.from_origin(
            west_edge,
            north_edge,
            x_res,
            y_res,
        )

        # from_origin() also assumes rows run north -> south. If the source
        # latitudes ascend, the rows must be flipped to match the transform.
        if era5_lats[1] > era5_lats[0]:
            stacked_data = stacked_data[:, ::-1, :]

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
                "latitude": target_lats,
                "longitude": target_lons,
            },
        )
        resampled_datasets.append(resampled_ds)

    # Concatenate all time steps back into one Xarray Dataset
    final_ds = xr.concat(resampled_datasets, dim="valid_time")
    return final_ds


# --- 3. Run and Save ---
era5_nc_path = "dataset/ERA5-Land/final-era5_rechunked.nc"
output_nc_path = "dataset/ERA5-Land/era5_resampled_1km_v2.nc"

final_era5_ds = resample_era5_to_1km(era5_nc_path)

# Save the final dataset with compression
encoding = {var: {"zlib": True, "complevel": 5} for var in final_era5_ds.data_vars}
final_era5_ds.to_netcdf(output_nc_path, format="NETCDF4", encoding=encoding)

print(f"\nFinal 1km resampled ERA5 data saved to: {output_nc_path}")
