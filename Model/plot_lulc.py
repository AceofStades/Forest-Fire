import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

file_path = 'dataset/LULC/UK_LULC50K_2016.tif'

try:
    with rasterio.open(file_path) as src:
        lulc_array = src.read(1)

        print(f"File Name: {src.name}")
        print(f"Number of bands: {src.count}")
        print(f"Dimensions (rows, cols): {src.shape}")
        print(f"CRS (Coordinate Reference System): {src.crs}")
        print(f"Transform: {src.transform}")

        colors = [
            '#008000',
            '#FFFF00',
            '#808080',
            '#0000FF',
            '#A52A2A',
            '#FFDAB9'
        ]

        cmap = ListedColormap(colors)

        plt.figure(figsize=(10, 10))
        plt.imshow(lulc_array, cmap=cmap)

        class_labels = {
            1: 'Forest',
            2: 'Agriculture',
            3: 'Urban',
            4: 'Water',
            5: 'Barren',
            6: 'Grassland'
        }

        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=colors[i-1], label=class_labels[i]) for i in range(1, len(class_labels) + 1)]
        plt.legend(handles=patches, title="LULC Classes", loc='upper left')

        plt.title('Land Use/Land Cover (LULC) Map')
        plt.xlabel('Column #')
        plt.ylabel('Row #')

        plt.savefig('lulc.png')
        print("Plot saved as lulc.png")

except rasterio.errors.RasterioIOError as e:
    print(f"Error reading the file: {e}")
    print("Please check if the file path is correct and the file exists.")
