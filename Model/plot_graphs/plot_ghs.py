import rasterio
import matplotlib.pyplot as plt

# Replace with the path to your main GeoTIFF file.
# Rasterio will automatically find and use the .ovr file if it exists.
file_path = 'dataset/GHS/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R6_C26.tif'

try:
    with rasterio.open(file_path) as src:
        # Instead of reading the full resolution data,
        # we'll read a downsampled version using the `out_shape` parameter.
        # This loads a much smaller array into memory.

        # Let's target a manageable size, for example, 1000x1000 pixels.
        out_shape = (1000, 1000)

        print(f"Reading a downsampled version of the raster with shape {out_shape}...")
        downsampled_data = src.read(
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.bilinear
        )

        print(f"Downsampled data shape: {downsampled_data.shape}")

        plt.figure(figsize=(10, 10))
        plt.imshow(downsampled_data[0], cmap='viridis', vmin=0, vmax=100)
        plt.colorbar(label='Built-up Surface (%)')
        plt.title('GHS Built-up Surface (Downsampled for Visualization)')

        # Save the plot to a file
        plt.savefig('ghs_downsampled_plot.png')
        print("Plot of downsampled data saved as ghs_downsampled_plot.png")

except rasterio.errors.RasterioIOError as e:
    print(f"Error reading the file: {e}")
    print("Please check if the file path is correct and the file exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
