import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from shapely.ops import transform
import pyproj
import matplotlib.pyplot as plt

# Define the file paths
input_file_path = 'dataset/GHS/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R6_C26.tif'
output_file_path = 'ghs_uttarakhand_clipped.tif'

try:
    # --- Step 1: Clip the large GeoTIFF to a smaller area ---

    # Define the bounding box for Uttarakhand (in decimal degrees)
    bbox_coords = [77.575, 28.715, 81.043, 31.467]
    geom = box(*bbox_coords)

    with rasterio.open(input_file_path) as src:
        print("Clipping the large GHS file to the Uttarakhand region...")

        # Define the source and destination CRSs for reprojection
        project = pyproj.Transformer.from_crs(
            'EPSG:4326',  # Source CRS of your bounding box
            src.crs,      # Destination CRS of the raster
            always_xy=True
        ).transform

        # Manually reproject the bounding box geometry
        geom_reprojected = transform(project, geom)

        # Perform the clipping using the reprojected geometry
        out_image, out_transform = mask(src, [geom_reprojected], crop=True)

        # Get a copy of the source metadata to update for the output file
        out_meta = src.meta.copy()

        # Update the metadata to reflect the new file dimensions, transform, and CRS
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs
        })

        # Save the clipped data to a new GeoTIFF file
        with rasterio.open(output_file_path, "w", **out_meta) as dest:
            # --- This is the key change ---
            # Write the entire 3D array, which contains the single band
            dest.write(out_image)

    print(f"Clipped GeoTIFF saved successfully to: {output_file_path}\n")

    # --- Step 2: Plot the smaller, clipped GeoTIFF ---

    print("Now plotting the clipped file...")
    with rasterio.open(output_file_path) as src_clipped:
        # The clipped file is small enough to be loaded into memory
        clipped_array = src_clipped.read(1)

        plt.figure(figsize=(10, 10))
        plt.imshow(clipped_array, cmap='viridis', vmin=0, vmax=100)
        plt.colorbar(label='Built-up Surface (%)')
        plt.title('GHS Built-up Surface (Clipped to Uttarakhand)')

        # Save the plot to a file
        plt.savefig('ghs_uttarakhand_plot.png')
        print("Plot of clipped data saved as ghs_uttarakhand_plot.png")

except rasterio.errors.RasterioIOError as e:
    print(f"Error reading the file: {e}")
    print("Please check if the file path is correct and the file exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
