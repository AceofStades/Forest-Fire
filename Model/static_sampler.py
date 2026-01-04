import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

# Use the same grid as your ERA5
target_crs = "EPSG:4326"
target_resolution = 0.009
target_bounds = rasterio.coords.BoundingBox(
    left=77.5, bottom=28.7, right=81.1, top=31.5
)
target_width, target_height = 400, 311  # Based on your validate output


def resample_fixed(input_path, output_path, is_categorical=False):
    with rasterio.open(input_path) as src:
        # Detect the source NoData value (Crucial!)
        nodata_val = src.nodata if src.nodata is not None else 0

        destination_array = np.empty((target_height, target_width), dtype=np.float32)

        reproject(
            source=src.read(1),
            destination=destination_array,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata_val,  # Tell it what to ignore
            dst_transform=rasterio.transform.from_bounds(
                *target_bounds, target_width, target_height
            ),
            dst_crs=target_crs,
            dst_nodata=0,  # Fill empty areas with 0 (consistent for U-Net)
            resampling=Resampling.nearest if is_categorical else Resampling.bilinear,
        )

        profile = src.profile
        profile.update(
            width=target_width,
            height=target_height,
            transform=rasterio.transform.from_bounds(
                *target_bounds, target_width, target_height
            ),
            crs=target_crs,
            nodata=0,
            dtype="float32",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(destination_array, 1)
    print(f"Fixed: {output_path}")


# Run the fix
resample_fixed("dataset/DEM/merged_dem.tif", "dataset/resampled-fix/dem_resampled.tif")
resample_fixed(
    "dataset/LULC/UK_LULC50K_2016.tif",
    "dataset/resampled-fix/lulc_resampled.tif",
    is_categorical=True,
)
resample_fixed(
    "dataset/GHS/ghs_uttarakhand_clipped.tif", "dataset/resampled-fix/ghs_resampled.tif"
)
