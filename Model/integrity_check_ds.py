import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer

# --- Paths ---
STACK_PATH = "dataset/final_feature_stack2.nc"
SOURCE_FILES = {
    "DEM": "dataset/DEM/merged_dem.tif",
    "LULC": "dataset/LULC/UK_LULC50K_2016.tif",
    "GHS_BUILT": "dataset/GHS/ghs_uttarakhand_clipped.tif",
}


def sample_source_raster(file_path, lon, lat):
    """Get the raw value from the source GeoTIFF at a specific lon/lat."""
    try:
        with rasterio.open(file_path) as src:
            # Transform lon/lat to the raster's internal CRS
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)

            # Sample the value
            for val in src.sample([(x, y)]):
                return val[0]
    except Exception:
        return np.nan


def run_integrity_check():
    print("--- Deep Integrity Check: Stack vs. Source ---")
    ds = xr.open_dataset(STACK_PATH)

    # We will check the first time step (static variables are constant anyway)
    sample_ts = ds.isel(valid_time=0)

    # 1. Coordinate Coverage Check
    lats = sample_ts.latitude.values
    lons = sample_ts.longitude.values

    print(f"\n[1] Checking Corner Pixels (Likely Zeros):")
    # Top-Left corner (usually where zeros appear if bounds are slightly off)
    test_lat, test_lon = lats[0], lons[0]

    results = []
    for var_name, source_path in SOURCE_FILES.items():
        stack_val = sample_ts[var_name].values[0, 0]  # Top-left pixel
        source_val = sample_source_raster(source_path, test_lon, test_lat)

        status = "MATCH" if np.isclose(stack_val, source_val, atol=1e-3) else "MISMATCH"
        if stack_val == 0 and (source_val != 0 and not np.isnan(source_val)):
            status = "!!! FAILED (Data Lost) !!!"

        results.append(
            {
                "Variable": var_name,
                "Stack_Val": stack_val,
                "Source_Val": source_val,
                "Status": status,
            }
        )

    print(pd.DataFrame(results))

    # 2. Global Zero Distribution
    print("\n[2] Spatial Zero Distribution (%) :")
    for var in SOURCE_FILES.keys():
        data = sample_ts[var].values
        total_pixels = data.size
        zero_pixels = np.count_nonzero(data == 0)
        zero_pct = (zero_pixels / total_pixels) * 100
        print(f"    - {var:10}: {zero_pct:6.2f}% of the map is 0.0")

    # 3. Edge Analysis
    # If zeros are only at the edges, it's a bounding box padding issue.
    # If zeros are in the middle, it's a source data issue.
    print("\n[3] Edge vs. Interior Analysis:")
    for var in ["DEM", "LULC"]:
        data = sample_ts[var].values
        edge_sum = (
            np.sum(data[0, :])
            + np.sum(data[-1, :])
            + np.sum(data[:, 0])
            + np.sum(data[:, -1])
        )
        if edge_sum == 0:
            print(
                f"    - {var}: Edge pixels are all ZERO. This suggests the stack is slightly wider than the source."
            )
        else:
            print(f"    - {var}: Edges contain data.")

    print("\n--- Check Finished ---")


if __name__ == "__main__":
    run_integrity_check()
