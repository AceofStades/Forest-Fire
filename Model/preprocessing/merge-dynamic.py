import os

import dask.diagnostics
import numpy as np
import pandas as pd
import rioxarray  # ensures GeoTIFFs can be opened by xarray
import xarray as xr
from tqdm import tqdm

# --- 1. Define File Paths ---
ERA5_NC_PATH = "dataset/ERA5-Land/era5_resampled_1km.nc"
DEM_PATH = "dataset/resampled-fix/dem_resampled.tif"
LULC_PATH = "dataset/resampled-fix/lulc_resampled.tif"
GHS_PATH = "dataset/resampled-fix/ghs_resampled.tif"

# CHANGED: Use CSV instead of static TIF for dynamic fire generation
MODIS_CSV_PATH = "dataset/MODIS/final-modis.csv"
OUTPUT_NC_PATH = "dataset/final_feature_stack_DYNAMIC.nc"


# --- 2. Dynamic Fire Generation ---
def generate_dynamic_fire_layer(grid_ds, csv_path):
    """
    Reads MODIS CSV and rasterizes fires into the time-space grid defined by grid_ds.
    grid_ds: The ERA5 dataset (already cropped to ROI) defining Time, Lat, Lon.
    """
    print(f"Generating Dynamic Fire Layer from {csv_path}...")

    # 1. Extract Grid Coordinates
    times = pd.to_datetime(grid_ds.valid_time.values)
    lats = grid_ds.latitude.values
    lons = grid_ds.longitude.values

    T, H, W = len(times), len(lats), len(lons)
    print(f"  Target Grid: {T} steps x {H} lat x {W} lon")

    # 2. Load and Clean CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"MODIS CSV not found at {csv_path}")

    df.columns = [c.lower() for c in df.columns]

    # 3. Parse Dates
    print("  Parsing fire dates...")
    # Handle standard MODIS formats (Date + HHMM Time)
    if "acq_date" in df.columns and "acq_time" in df.columns:
        df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
        df["dt_str"] = df["acq_date"] + " " + df["acq_time"]
        df["datetime"] = pd.to_datetime(df["dt_str"], format="%Y-%m-%d %H%M")
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("MODIS CSV missing 'acq_date'/'acq_time' or 'date' columns.")

    # 4. Filter to Grid Bounds
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # Filter bounds
    df = df[(df.latitude >= lat_min) & (df.latitude <= lat_max)]
    df = df[(df.longitude >= lon_min) & (df.longitude <= lon_max)]

    # Filter time (include buffer)
    df = df[(df.datetime >= times[0]) & (df.datetime <= times[-1])]

    print(f"  Fires matched within bounds: {len(df)}")

    if len(df) == 0:
        print(
            "  WARNING: No fires found in the active region/time! Returning empty mask."
        )
        fire_grid = np.zeros((T, H, W), dtype=np.float32)
    else:
        # 5. Rasterize (Vectorized)
        fire_grid = np.zeros((T, H, W), dtype=np.float32)

        # Lat/Lon Steps
        lat_step = abs(lats[1] - lats[0])
        lon_step = abs(lons[1] - lons[0])

        # Map Lat/Lon to Indices
        # Check if Lat is descending (standard) or ascending
        if lats[0] > lats[1]:  # Descending
            y_indices = ((lats[0] - df.latitude.values) / lat_step).astype(int)
        else:  # Ascending
            y_indices = ((df.latitude.values - lats[0]) / lat_step).astype(int)

        x_indices = ((df.longitude.values - lons[0]) / lon_step).astype(int)

        # Map Time to Indices (Nearest Neighbor)
        nc_times_int = times.astype(np.int64)
        csv_times_int = df.datetime.values.astype(np.int64)
        t_indices = np.searchsorted(nc_times_int, csv_times_int)

        # Clip indices to safe bounds
        y_indices = np.clip(y_indices, 0, H - 1)
        x_indices = np.clip(x_indices, 0, W - 1)
        t_indices = np.clip(t_indices, 0, T - 1)

        # Fill grid
        # Using numpy advanced indexing to set 1s
        print("  Filling 3D Time-Space Matrix...")
        fire_grid[t_indices, y_indices, x_indices] = 1.0

    print(f"  Total Fire Pixels: {fire_grid.sum()}")

    # 6. Create DataArray
    da_fire = xr.DataArray(
        data=fire_grid,
        dims=["valid_time", "latitude", "longitude"],
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
        name="MODIS_FIRE_T1",
    )

    return da_fire


# --- 3. Master Alignment and Loading ---
def prepare_static_layers(era5_template_ds):
    print("Loading and cleaning static GeoTIFF datasets...")

    def load_and_clean_static(path, name, interp_method):
        # Open using rioxarray
        da = rioxarray.open_rasterio(path, masked=True).sel(band=1, drop=True)
        da = da.rename({"x": "longitude", "y": "latitude"})
        da.name = name

        # FIX: Specific cleaning for GHS_BUILT
        if name == "GHS_BUILT":
            da = da.where(da != 255, 0)

        if "spatial_ref" in da.coords:
            da = da.drop_vars("spatial_ref")

        print(f"  Aligning {name} using '{interp_method}' interpolation...")
        # Initial alignment to the template grid
        da_aligned = da.interp_like(era5_template_ds, method=interp_method)
        return da_aligned.fillna(0)

    # Load static layers
    dem = load_and_clean_static(DEM_PATH, "DEM", "linear")
    lulc = load_and_clean_static(LULC_PATH, "LULC", "nearest")
    ghs = load_and_clean_static(GHS_PATH, "GHS_BUILT", "linear")

    # --- THE MASTER CROP: Remove zero-padding ---
    print("Calculating intersection mask to remove edge padding...")
    valid_mask = (dem > 0) & (dem.notnull())

    # Find coordinates where data is valid
    valid_coords = np.argwhere(valid_mask.values)
    lat_min, lat_max = valid_coords[:, 0].min(), valid_coords[:, 0].max()
    lon_min, lon_max = valid_coords[:, 1].min(), valid_coords[:, 1].max()

    print(
        f"  Cropping bounds: Lat_idx[{lat_min}:{lat_max}], Lon_idx[{lon_min}:{lon_max}]"
    )

    # Slice the static layers
    dem = dem.isel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    lulc = lulc.isel(
        latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
    )
    ghs = ghs.isel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    # Merge static layers
    static_ds = xr.merge(
        [
            dem.astype(np.float32),
            lulc.astype(np.int16),
            ghs.astype(np.float32),
        ]
    )

    return static_ds, (lat_min, lat_max, lon_min, lon_max)


# --- 4. Final Fusion and Save ---
def fuse_master_stack(output_path):
    print("Loading ERA5 time-series data...")
    era5_ds = xr.open_dataset(ERA5_NC_PATH, chunks={"valid_time": 50})
    if "spatial_ref" in era5_ds.coords:
        era5_ds = era5_ds.drop_vars("spatial_ref")

    # 1. Prepare Static Layers & Crop Bounds
    static_ds, slices = prepare_static_layers(era5_ds)
    lat_min, lat_max, lon_min, lon_max = slices

    print("Merging features and applying master crop to weather data...")
    # 2. Crop ERA5 to match
    era5_cropped = era5_ds.isel(
        latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
    )

    # 3. Generate Dynamic Fire Layer
    # Pass the CROPPED ERA5 template so fire matches the final grid exactly
    fire_da = generate_dynamic_fire_layer(era5_cropped, MODIS_CSV_PATH)

    # 4. Perform Final Merge
    print("Merging Weather, Static Maps, and Dynamic Fire...")
    final_ds = xr.merge([era5_cropped, static_ds, fire_da])

    # FORCE DIMENSION ORDER
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
