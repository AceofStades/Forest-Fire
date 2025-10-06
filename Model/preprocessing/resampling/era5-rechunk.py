import xarray as xr
import os
import dask.diagnostics
import tqdm  # tqdm is usually installed with dask and xarray

input_path = "dataset/ERA5-Land/final-era5.nc"
output_path = "dataset/ERA5-Land/final-era5_rechunked.nc"

# --- 1. Load the original merged file ---
print(f"Loading original file: {input_path}")
ds = xr.open_dataset(input_path)

# --- 2. Define the new chunking strategy ---
# The total number of valid_time steps should be 1464 (April and May)
chunks_to_use = {
    "valid_time": 1,  # Critical for parallel processing (reading one time step at a time)
    "latitude": ds.sizes["latitude"] // 4,
    "longitude": ds.sizes["longitude"] // 4,
}
print(f"Applying new chunking strategy: {chunks_to_use}")

# --- 3. Rechunk and save the new file with a progress bar ---
# Rechunking creates a Dask graph, but doesn't execute it yet.
ds_rechunked = ds.chunk(chunks_to_use)

# Create an encoding dictionary for compression for all data variables
encoding = {var: {"zlib": True, "complevel": 5} for var in ds_rechunked.data_vars}

print("\nStarting rechunking and saving process...")

# The 'dask.diagnostics.ProgressBar' is enabled when executing the final output.
with dask.diagnostics.ProgressBar():
    # Save the new file using the 'encoding' parameter
    ds_rechunked.to_netcdf(output_path, format="NETCDF4", encoding=encoding)

print(f"\nSuccessfully rechunked and saved optimized file to: {output_path}")
