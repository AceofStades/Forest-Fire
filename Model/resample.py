import pandas as pd
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import numpy as np
import os
from datetime import datetime
from shapely.geometry import Point

# --- 1. Define the Target Grid ---
# We will use a 30m resolution and the WGS84 CRS (EPSG:4326)
target_crs = 'EPSG:4326'
target_resolution = 0.0002777777777777778  # ~30 meters in decimal degrees

# Bounding box for Uttarakhand
target_bounds = rasterio.coords.BoundingBox(
    left=77.5,
    bottom=28.7,
    right=81.1,
    top=31.5
)

# Calculate the dimensions of the target grid
target_width = int(round((target_bounds.right - target_bounds.left) / target_resolution))
target_height = int(round((target_bounds.top - target_bounds.bottom) / target_resolution))

print(f"Target Grid Dimensions: {target_width} x {target_height} pixels")
print(f"Target Resolution: {target_resolution} decimal degrees\n")

# --- 2. Resample GeoTIFFs ---
def resample_geotiff(input_path, output_path):
    with rasterio.open(input_path) as src:
        source_array = src.read(1)
        source_transform = src.transform
        source_crs = src.crs

        destination_array = np.empty((target_height, target_width), dtype=src.meta['dtype'])

        reproject(
            source=source_array,
            destination=destination_array,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=rasterio.transform.from_bounds(
                target_bounds.left, target_bounds.bottom, target_bounds.right, target_bounds.top,
                target_width, target_height
            ),
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

        profile = src.profile
        profile.update(
            width=target_width,
            height=target_height,
            transform=rasterio.transform.from_bounds(
                target_bounds.left, target_bounds.bottom, target_bounds.right, target_bounds.top,
                target_width, target_height
            ),
            crs=target_crs
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(destination_array.astype(profile['dtype']), 1)
        print(f"Successfully resampled and saved: {output_path}")

# --- Replace these with your actual file paths ---
dem_path = 'dataset/DEM/merged_dem.tif'
lulc_path = 'dataset/LULC/UK_LULC50K_2016.tif'
ghs_path = 'dataset/GHS/ghs_uttarakhand_clipped.tif'

print("Resampling GeoTIFF files...")
resample_geotiff(dem_path, 'dem_resampled.tif')
resample_geotiff(lulc_path, 'lulc_resampled.tif')
resample_geotiff(ghs_path, 'ghs_resampled.tif')

# --- 3. Rasterize MODIS (.csv) ---
def rasterize_modis_data(csv_path, output_path):
    print("\nRasterizing MODIS data...")
    df = pd.read_csv(csv_path)

    # Use shapely to create point geometries
    modis_points = [Point(row.longitude, row.latitude) for _, row in df.iterrows()]

    # Create a destination array with the target grid dimensions
    modis_raster = np.zeros((target_height, target_width), dtype=np.uint8)

    # Rasterize the points onto the empty grid
    shapes = [(point, 1) for point in modis_points]
    modis_raster = rasterize(
        shapes=shapes,
        out_shape=(target_height, target_width),
        transform=rasterio.transform.from_bounds(
            target_bounds.left, target_bounds.bottom, target_bounds.right, target_bounds.top,
            target_width, target_height
        ),
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )

    # Save the raster to a GeoTIFF
    profile = {
        'driver': 'GTiff',
        'height': target_height,
        'width': target_width,
        'count': 1,
        'dtype': modis_raster.dtype,
        'crs': target_crs,
        'transform': rasterio.transform.from_bounds(
            target_bounds.left, target_bounds.bottom, target_bounds.right, target_bounds.top,
            target_width, target_height
        )
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(modis_raster, 1)
    print(f"Successfully rasterized MODIS data and saved as: {output_path}")

# Replace with your actual file path
modis_csv_path = 'dataset/MODIS/final-modis.csv'
rasterize_modis_data(modis_csv_path, 'modis_raster.tif')

# --- 4. Resample ERA5 (.nc) ---
def resample_era5(input_nc_path, output_folder):
    print("\nResampling ERA5 data...")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ds = xr.open_dataset(input_nc_path)

    for i, valid_time in enumerate(ds['valid_time'].values):
        print(f"Processing time step {i+1}/{len(ds['valid_time'])}: {valid_time}")

        timestamp = pd.to_datetime(str(valid_time)).strftime('%Y%m%d%H%M')

        data_at_time = ds.isel(valid_time=i).drop_vars('valid_time')

        variables_to_stack = list(data_at_time.data_vars)
        stacked_data = np.stack([data_at_time[var].values for var in variables_to_stack], axis=0)

        x_res = abs(data_at_time.coords['longitude'].values[1] - data_at_time.coords['longitude'].values[0])
        y_res = abs(data_at_time.coords['latitude'].values[1] - data_at_time.coords['latitude'].values[0])
        era5_transform = rasterio.transform.from_origin(
            data_at_time.coords['longitude'].values[0],
            data_at_time.coords['latitude'].values[0],
            x_res,
            y_res
        )

        output_data = np.empty((len(variables_to_stack), target_height, target_width), dtype=np.float32)

        reproject(
            source=stacked_data,
            destination=output_data,
            src_transform=era5_transform,
            src_crs='EPSG:4326',
            dst_transform=rasterio.transform.from_bounds(
                target_bounds.left, target_bounds.bottom, target_bounds.right, target_bounds.top,
                target_width, target_height
            ),
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

        profile = {
            'driver': 'GTiff',
            'height': target_height,
            'width': target_width,
            'count': len(variables_to_stack),
            'dtype': output_data.dtype,
            'crs': target_crs,
            'transform': rasterio.transform.from_bounds(
                target_bounds.left, target_bounds.bottom, target_bounds.right, target_bounds.top,
                target_width, target_height
            )
        }

        with rasterio.open(f'{output_folder}/era5_{timestamp}.tif', 'w', **profile) as dst:
            for i in range(len(variables_to_stack)):
                dst.write(output_data[i].astype(profile['dtype']), i + 1)
            dst.descriptions = variables_to_stack

    print("ERA5 resampling complete.")

# Replace with your actual ERA5 file path
era5_nc_path = 'dataset/ERA5-Land/final-era5.nc'
resample_era5(era5_nc_path, 'era5_resampled')
