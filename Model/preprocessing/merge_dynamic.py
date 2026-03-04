import os

import dask.diagnostics
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from tqdm import tqdm

# --- 1. Paths ---
ERA5_NC_PATH = "dataset/ERA5-Land/era5_resampled_1km.nc"
DEM_PATH = "dataset/resampled-fix/dem_resampled.tif"
LULC_PATH = "dataset/resampled-fix/lulc_resampled.tif"
GHS_PATH = "dataset/resampled-fix/ghs_resampled.tif"
MODIS_CSV_PATH = "dataset/MODIS/final-modis.csv"
OUTPUT_NC_PATH = "dataset/final_feature_stack_DYNAMIC_new.nc"


def load_and_clean_static(path, name, template_ds, interp_method="nearest"):
    """Loads a static GeoTIFF, cleans it, and aligns it to the template grid."""
    print(f"Processing {name} from {path}...")
    try:
        da = rioxarray.open_rasterio(path, masked=True).sel(band=1, drop=True)
        da = da.rename({"x": "longitude", "y": "latitude"})
        da.name = name

        # FIX: GHS_BUILT 255 -> 0
        if name == "GHS_BUILT":
            da = da.where(da != 255, 0)

        if "spatial_ref" in da.coords:
            da = da.drop_vars("spatial_ref")

        # Align to template
        da_aligned = da.interp_like(template_ds, method=interp_method)
        return da_aligned.fillna(0)
    except Exception as e:
        print(f"Error loading {name}: {e}")
        raise


def generate_dynamic_fire_mask(csv_path, template_ds):
    """
    Generates a 3D (Time, Lat, Lon) binary fire mask from MODIS CSV points.
    """
    print(f"Generating Dynamic Fire Mask from {csv_path}...")

    # 1. Load CSV
    df = pd.read_csv(csv_path)
    # Parse timestamps
    # Assuming 'acq_timestamp' exists and is in a standard format
    # If not, construct from date and time
    if "acq_timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["acq_timestamp"])
    else:
        # Fallback: construct from acq_date + acq_time
        # acq_time is usually HHMM as integer. Need to pad.
        df["acq_time_str"] = df["acq_time"].astype(str).str.zfill(4)
        df["timestamp"] = pd.to_datetime(
            df["acq_date"] + " " + df["acq_time_str"], format="%Y-%m-%d %H%M"
        )

    # 2. Prepare Empty 3D Array matches ERA5 dimensions
    times = template_ds.valid_time.values
    lats = template_ds.latitude.values
    lons = template_ds.longitude.values

    fire_mask = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)

    # 3. Time Mapping
    print("Mapping fire events to nearest ERA5 hourly slot...")
    df["nearest_hour"] = df["timestamp"].dt.round("h")

    # Filter out points outside our time range
    # Ensure timezone naivety for comparison
    time_min = pd.to_datetime(times[0]).tz_localize(None)
    time_max = pd.to_datetime(times[-1]).tz_localize(None)

    # Ensure DF is also naive
    df["nearest_hour"] = df["nearest_hour"].dt.tz_localize(None)

    initial_count = len(df)
    df = df[(df["nearest_hour"] >= time_min) & (df["nearest_hour"] <= time_max)]
    filtered_count = len(df)
    print(f"  Points in time range: {filtered_count} / {initial_count}")

    # Create a lookup for time indices
    # Convert numpy datetime64 to pandas timestamp for robust lookup
    time_index_map = {
        pd.to_datetime(t).tz_localize(None): i for i, t in enumerate(times)
    }

    # Group by time step for efficiency
    grouped = df.groupby("nearest_hour")

    mapped_pixels = 0

    print("Rasterizing fire points...")
    for time_val, group in tqdm(grouped, desc="Processing Time Steps"):
        if time_val not in time_index_map:
            continue

        t_idx = time_index_map[time_val]

        # Get coordinates
        p_lats = group["latitude"].values
        p_lons = group["longitude"].values

        # Map to indices
        # We find the nearest lat/lon index for each point
        # abs(lat_grid - lat_point).argmin()

        # Optimization: broadcasting might be too heavy if grid is large.
        # But for 300x400 grid it is fine? No, 1km grid is large.
        # Better to use searchsorted or digitize.

        # Assumption: lats/lons are sorted.
        # Check sort order
        if lats[1] > lats[0]:  # Ascending
            lat_idxs = np.searchsorted(lats, p_lats)
        else:  # Descending
            # searchsorted requires ascending
            lat_idxs = len(lats) - 1 - np.searchsorted(lats[::-1], p_lats)

        if lons[1] > lons[0]:  # Ascending
            lon_idxs = np.searchsorted(lons, p_lons)
        else:  # Descending
            lon_idxs = len(lons) - 1 - np.searchsorted(lons[::-1], p_lons)

        # Clip to bounds
        lat_idxs = np.clip(lat_idxs, 0, len(lats) - 1)
        lon_idxs = np.clip(lon_idxs, 0, len(lons) - 1)

        # Mark fire
        # We can use advanced indexing
        fire_mask[t_idx, lat_idxs, lon_idxs] = 1.0
        mapped_pixels += len(lat_idxs)

    print(f"Total fire pixels mapped: {mapped_pixels}")

    # Create DataArray
    fire_da = xr.DataArray(
        fire_mask,
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
        dims=("valid_time", "latitude", "longitude"),
        name="MODIS_FIRE_T1",
    )

    return fire_da


def main():
    print(f"--- Starting Dynamic Merge (Output: {OUTPUT_NC_PATH}) ---")

    # 1. Load ERA5 Template (Time Series)
    print(f"Loading ERA5 from {ERA5_NC_PATH}...")
    era5_ds = xr.open_dataset(ERA5_NC_PATH, chunks={"valid_time": 50})
    if "spatial_ref" in era5_ds.coords:
        era5_ds = era5_ds.drop_vars("spatial_ref")

    # 2. Load Static Layers
    dem = load_and_clean_static(DEM_PATH, "DEM", era5_ds, "linear")
    lulc = load_and_clean_static(LULC_PATH, "LULC", era5_ds, "nearest")
    ghs = load_and_clean_static(GHS_PATH, "GHS_BUILT", era5_ds, "linear")

    # 3. Generate Dynamic Fire Mask
    fire_da = generate_dynamic_fire_mask(MODIS_CSV_PATH, era5_ds)

    # 4. Crop to Valid Data Area (Using DEM)
    print("Calculating Valid Data Crop...")
    # DEM is 2D (Lat, Lon), fire is 3D (Time, Lat, Lon).
    # We use DEM to determine spatial bounds.
    valid_mask = (dem > 0) & (dem.notnull())
    valid_coords = np.argwhere(valid_mask.values)

    if len(valid_coords) == 0:
        print("Error: No valid data found in DEM. Check source files.")
        return

    lat_min_idx = valid_coords[:, 0].min()
    lat_max_idx = valid_coords[:, 0].max()
    lon_min_idx = valid_coords[:, 1].min()
    lon_max_idx = valid_coords[:, 1].max()

    print(
        f"  Cropping Lat Index: {lat_min_idx}-{lat_max_idx}, Lon Index: {lon_min_idx}-{lon_max_idx}"
    )

    # Apply Crop
    def crop(da):
        return da.isel(
            latitude=slice(lat_min_idx, lat_max_idx),
            longitude=slice(lon_min_idx, lon_max_idx),
        )

    print("Cropping datasets...")
    era5_cropped = crop(era5_ds)
    dem_cropped = crop(dem)
    lulc_cropped = crop(lulc)
    ghs_cropped = crop(ghs)
    fire_cropped = crop(fire_da)

    # 5. Merge
    print("Merging datasets...")

    # Create the final Dataset
    # ERA5 is the base. We add static layers as variables (coordinates?)
    # Usually static layers are just variables with (lat, lon) dims, not (time, lat, lon).
    # Xarray handles this fine.

    # However, 'fire_cropped' IS 3D (time, lat, lon).

    final_ds = xr.merge(
        [
            era5_cropped,
            fire_cropped,
            dem_cropped.astype(np.float32),
            lulc_cropped.astype(np.int16),
            ghs_cropped.astype(np.float32),
        ]
    )

    # Force Dimension Order
    final_ds = final_ds.transpose("valid_time", "latitude", "longitude")

    # 6. Save
    print(f"Saving to {OUTPUT_NC_PATH}...")

    # Compression
    encoding = {var: {"zlib": True, "complevel": 5} for var in final_ds.data_vars}

    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_NC_PATH), exist_ok=True)

    if os.path.exists(OUTPUT_NC_PATH):
        os.remove(OUTPUT_NC_PATH)

    with dask.diagnostics.ProgressBar():
        final_ds.to_netcdf(OUTPUT_NC_PATH, format="netcdf4", encoding=encoding)

    print("✅ Done! Dynamic dataset created.")


if __name__ == "__main__":
    main()
