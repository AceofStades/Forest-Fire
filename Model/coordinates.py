import rasterio

# Define a function to get and print the coordinates of a GeoTIFF file
def get_geotiff_coordinates(file_path):
    """
    Opens a GeoTIFF file, extracts its bounding box, and prints the coordinates.
    The coordinates are converted to decimal degrees (EPSG:4326) for easy interpretation.

    Args:
        file_path (str): The path to the GeoTIFF file.
    """
    try:
        with rasterio.open(file_path) as src:
            # The bounds object contains the bounding box coordinates in the file's native CRS
            bounds = src.bounds

            # Get the file's CRS (Coordinate Reference System)
            file_crs = src.crs

            # Print the native bounds and CRS for context
            print(f"File Name: {file_path}")
            print(f"Native Bounds ({file_crs}):")
            print(f"  Left (minx): {bounds.left}")
            print(f"  Bottom (miny): {bounds.bottom}")
            print(f"  Right (maxx): {bounds.right}")
            print(f"  Top (maxy): {bounds.top}")
            print("-" * 30)

            # Check if the file's CRS is already EPSG:4326 (decimal degrees)
            if file_crs.to_epsg() == 4326:
                # If so, the native bounds are already in decimal degrees
                print("Bounds in Decimal Degrees (EPSG:4326):")
                print(f"  West (min_lon): {bounds.left}")
                print(f"  South (min_lat): {bounds.bottom}")
                print(f"  East (max_lon): {bounds.right}")
                print(f"  North (max_lat): {bounds.top}")
            else:
                # If not, convert the bounds to EPSG:4326 for a universal format
                from rasterio.warp import transform_bounds

                # Transform the bounds from the file's CRS to EPSG:4326
                transformed_bounds = transform_bounds(file_crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top)

                print("Bounds in Decimal Degrees (EPSG:4326):")
                print(f"  West (min_lon): {transformed_bounds[0]}")
                print(f"  South (min_lat): {transformed_bounds[1]}")
                print(f"  East (max_lon): {transformed_bounds[2]}")
                print(f"  North (max_lat): {transformed_bounds[3]}")

    except rasterio.errors.RasterioIOError as e:
        print(f"Error reading the file: {e}")
        print("Please check if the file path is correct and the file exists.")

# Example usage:
# Call the function for the files you want to check.
# Replace the paths below with the correct paths to your GeoTIFF files.
get_geotiff_coordinates('dataset/GHS/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R6_C26.tif')
print()
get_geotiff_coordinates('dataset/GHS/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R6_C27.tif')
print()
get_geotiff_coordinates('dataset/GHS/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R7_C26.tif')
