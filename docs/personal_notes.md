# Personal Research Notes and Context

## Project Attributes Table

### ERA5-Land (Weather/Climate Data)

| Attr | FullForm | Description |
|---|---|---|
| e | Evaporation | Total evaporation |
| tp | Total Precipitation | Total precipitation |
| tcc | Total Cloud Cover | Total cloud cover |
| stl1 | Soil Temperature Level 1 | Soil temperature at the top layer |
| slt | Soil Type | Type of soil |
| swvl1 | Volumetric Soil Water Level 1 | Soil moisture content |
| cvh | High Vegetation Cover | - |
| cvl | Low Vegetation Cover | - |
| tvh | Type of High Vegetation | - |
| tvl | Type of Low Vegetation | - |
| u10 | 10m U Component of Wind | Wind speed (East-West) |
| v10 | 10m V Componenet of Wind | Wind speed (North-South) |
| d2m | 2m Dewpoint Temperature | - |
| t2m | 2m Temperature | Ambient temperature |

**Geographical Bounding Box (Cropped for Northern India / Uttarakhand region):**
- North Latitude: 31.5° N
- South Latitude: 28.7° N
- West Longitude: 77.5° E
- East Longitude: 81.1° E

### MODIS (Historical Fire Data)

| Attr | Description |
|---|---|
| scan, track | These values relate to the dimensions of the fire pixel. They can be used to calculate the actual size of the pixel on the ground. |
| acq_date, acq_time | The date and time when the satellite detected the fire. Essential for temporal analysis and linking fire events to specific weather data. |
| satellite, instrument | The satellite (Terra or Aqua) and instrument (MODIS) that captured the data. |
| confidence | A value from 0-100% indicating the quality of the detection. High-confidence detections are more reliable. |
| bright_t31 | The brightness temperature of the fire pixel in band 31 (11.0µm), in Kelvin. Used for fire detection and characterization. |
| frp | Fire Radiative Power, measured in megawatts. A measure of the fire's energy and intensity, which can be used to estimate biomass burning. |
| daynight | Indicates whether the detection was made during the day or night. |
| type | Categorizes the type of fire, such as "1" for a vegetation fire. |


## Technical Concepts

### Convolution 3x3 layer
- A 3x3 convolution layer uses a 3x3 filter (or kernel) to slide across an input image or feature map, performing a dot product at each position to extract features like edges or textures.
- This process generates feature maps, which highlight specific patterns and are used in tasks like image recognition and object detection.
- The 3x3 size is a balance between capturing contextual information and computational efficiency, as multiple 3x3 convolutions can form a larger effective receptive field with fewer parameters than a single larger kernel.

### Max Pooling Layer
- A max pooling layer reduces the spatial size of an image or feature map by dividing it into a grid of smaller regions and selecting the maximum value from each region to form a new, compressed representation.
- This downsampling process makes the network more efficient by decreasing the computational load and memory requirements, helps prevent overfitting, and provides a degree of translation invariance, allowing the network to recognize features regardless of their exact position.

### Evaluation Metrics
**1. Performance Curves (`_curves.png`)**
- **Left: ROC Curve (Receiver Operating Characteristic)**
  - **X-axis (False Positive Rate):** Proportion of negative pixels (no fire) incorrectly classified as positive (fire).
  - **Y-axis (True Positive Rate / Recall):** Proportion of actual fire pixels correctly identified by the model.
  - **Interpretation:** A perfect model would go straight up to the top-left corner (0,1). The diagonal line represents a random guess. AUC (Area Under Curve) summarizes this; 1.0 is perfect, 0.5 is random.
- **Right: Precision-Recall Curve**
  - **X-axis (Recall):** How many of the actual fires did we find?
  - **Y-axis (Precision):** Of all the fires we predicted, how many were actually fires?
  - **Interpretation:** Crucial for imbalanced datasets like forest fires. A high curve towards the top-right corner is better. Shows the trade-off: catching more fires (high recall) usually means accepting more false alarms (lower precision).

**2. Visual Sample (`_visual.png`)**
- Qualitative "sanity check" of the model's performance on a single example.
- **Left (Input Channel 0):** Visualizes one input feature (usually temperature or a moisture index) to give context.
- **Middle (Target Fire):** The "Ground Truth". Yellow/orange areas are where the fire actually occurred.
- **Right (Prediction):** What the model predicted. Brighter/hotter colors indicate a higher probability of fire. Does the prediction shape match the actual fire?


## Data Sources & Links
- [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries?tab=download)
- [DEM (Bhuvan/NRSC)](https://bhoonidhi.nrsc.gov.in/)
- [LULC (Bhuvan/NRSC)](https://bhuvan-app1.nrsc.gov.in/thematic/thematic/index.php)
- [MODIS/VIIRS (FIRMS)](https://firms.modaps.eosdis.nasa.gov/country/)
- [FSI (Forest Survey of India)](https://www.fsiforestfire.gov.in/)
- [GHSL (Global Human Settlement Layer)](https://human-settlement.emergency.copernicus.eu/download.php)
