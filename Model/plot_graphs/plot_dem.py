import rasterio
import matplotlib.pyplot as plt

# Replace 'path/to/your/dem.tif' with the actual path to your downloaded DEM file
file_path = 'dataset/DEM/merged_dem.tif'

try:
    with rasterio.open(file_path) as src:
        dem_array = src.read(1)

        print(f"File Name: {src.name}")
        print(f"Number of bands: {src.count}")
        print(f"Dimensions (rows, cols): {src.shape}")
        print(f"CRS (Coordinate Reference System): {src.crs}")
        print(f"Transform: {src.transform}")

        plt.figure(figsize=(10, 10))
        plt.imshow(dem_array, cmap='terrain')
        plt.colorbar(label='Elevation (meters)')
        plt.title('Digital Elevation Model (DEM)')
        plt.xlabel('Column #')
        plt.ylabel('Row #')

        # --- Change this line ---
        # Save the plot as a PNG file instead of showing it
        plt.savefig('lulc.png')
        print("Plot saved as dem_plot.png")
        # ------------------------

except rasterio.errors.RasterioIOError as e:
    print(f"Error reading the file: {e}")
    print("Please check if the file path is correct and the file exists.")
