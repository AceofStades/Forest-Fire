import pandas as pd
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import multiprocessing as mp
from datetime import datetime

# --- 1. Define the Target Grid ---
target_crs = "EPSG:4326"
target_resolution = 0.0002777777777777778
target_bounds = rasterio.coords.BoundingBox(
    left=77.5, bottom=28.7, right=81.1, top=31.5
)
target_width = int(
    round((target_bounds.right - target_bounds.left) / target_resolution)
)
target_height = int(
    round((target_bounds.top - target_bounds.bottom) / target_resolution)
)

print(f"Target Grid Dimensions: {target_width} x {target_height} pixels")
print(f"Target Resolution: {target_resolution} decimal degrees\n")


# --- 2. Worker Function for Parallel Processing ---
def process_era5_timestep_safe(args):
    """
    Worker function to safely resample a single time step of ERA5 data.
    This function opens the file locally, reducing RAM usage.
    """
    input_nc_path, i, output_folder, num_timesteps = args

    try:
        # Open the dataset here in the worker process
        with xr.open_dataset(input_nc_path, chunks={"valid_time": 1}) as ds:
            # Select only the chunk of data this worker needs
            ds_chunk = ds.isel(valid_time=i)

            valid_time = ds_chunk["valid_time"].values
            timestamp = pd.to_datetime(str(valid_time)).strftime("%Y%m%d%H%M")
            data_at_time = ds_chunk.drop_vars("valid_time")

            variables_to_stack = list(data_at_time.data_vars)
            stacked_data = np.stack(
                [data_at_time[var].values for var in variables_to_stack], axis=0
            )

            x_res = abs(
                data_at_time.coords["longitude"].values[1]
                - data_at_time.coords["longitude"].values[0]
            )
            y_res = abs(
                data_at_time.coords["latitude"].values[1]
                - data_at_time.coords["latitude"].values[0]
            )
            era5_transform = rasterio.transform.from_origin(
                data_at_time.coords["longitude"].values[0],
                data_at_time.coords["latitude"].values[0],
                x_res,
                y_res,
            )

            output_data = np.empty(
                (len(variables_to_stack), target_height, target_width), dtype=np.float32
            )

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

            profile = {
                "driver": "GTiff",
                "height": target_height,
                "width": target_width,
                "count": len(variables_to_stack),
                "dtype": output_data.dtype,
                "crs": target_crs,
                "transform": rasterio.transform.from_bounds(
                    target_bounds.left,
                    target_bounds.bottom,
                    target_bounds.right,
                    target_bounds.top,
                    target_width,
                    target_height,
                ),
            }

            output_path = os.path.join(output_folder, f"era5_{timestamp}.tif")
            with rasterio.open(output_path, "w", **profile) as dst:
                for j in range(len(variables_to_stack)):
                    dst.write(output_data[j].astype(profile["dtype"]), j + 1)
                dst.descriptions = variables_to_stack

        return f"Processed time step {i + 1}/{num_timesteps}: {valid_time}"

    except Exception as e:
        return f"Error processing time step {i + 1}: {e}"


# --- 3. Main Function to Call Parallel Processing ---
def resample_era5_parallel_final(input_nc_path, output_folder, num_cores=4):
    print("\nStarting parallel resampling of ERA5 data with a safe memory approach...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_timesteps = xr.open_dataset(input_nc_path).dims["valid_time"]
    args_list = [
        (input_nc_path, i, output_folder, num_timesteps) for i in range(num_timesteps)
    ]

    print(f"Using {num_cores} cores for processing...")

    def log_progress(result):
        if result.startswith("Error"):
            print(f"Error: {result}")
        else:
            print(result)

    with mp.Pool(num_cores) as pool:
        for args in args_list:
            pool.apply_async(
                process_era5_timestep_final, (args,), callback=log_progress
            )

        pool.close()
        pool.join()

    print("\nERA5 parallel resampling complete.")


# --- Replace with your actual ERA5 file path ---
era5_nc_path = "dataset/ERA5-Land/final-era5.nc"
resample_era5_parallel_final(era5_nc_path, "era5_resampled")
