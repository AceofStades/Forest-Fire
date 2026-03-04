# Attributes Table

## ERA5-Land

|Attr|FullForm|
|---|---|
|e|Evaporation|
|tp|Total Precipitation|
|---|---|
|tcc|Total Cloud Cover|
|stl1|Soil Temperature Level 1|
|slt|Soil Type|
|swvl1|Volumetric Soil Water Level 1|
|cvh|High Vegetation Cover|
|cvl|Low Vegetation Cover|
|tvh|Type of High Vegetation|
|tvl|Type of Low Vegetation|
|u10|10m U Component of Wind|
|v10|10m V Componenet of Wind|
|d2m|2m Dewpoint Temperature|
|t2m|2m Temperature|

2m dewpoint temperature
2m temperature
Volumetric soil water layer 1
Total evaporation
10m u-component of wind
10m v-component of wind
Total precipitation
Low vegetation cover

North Latitude: 31.5° N
South Latitude: 28.7° N
West Longitude: 77.5° E
East Longitude: 81.1° E

## MODIS (Historical Fire Data)

|Attr|Description|
|---|---|
|scan, track|These values relate to the dimensions of the fire pixel. They can be used to calculate the actual size of the pixel on the ground.|
|acq_date, acq_time|The date and time when the satellite detected the fire. This is essential for temporal analysis and linking fire events to specific weather data.|
|satellite, instrument|The satellite (Terra or Aqua) and instrument (MODIS) that captured the data.|
|confidence|A value from 0-100% indicating the quality of the detection. High-confidence detections are more reliable.|
|bright_t31|The brightness temperature of the fire pixel in band 31 (11.0µm), in Kelvin. This is used for fire detection and characterization.|
|frp|Fire Radiative Power, measured in megawatts. This is a measure of the fire's energy and intensity, which can be used to estimate biomass burning.|
|daynight|Indicates whether the detection was made during the day or night.|
|type|Categorizes the type of fire, such as "1" for a vegetation fire.|

### Convolution 3x3 layer

- A 3x3 convolution layer uses a 3x3 filter (or kernel) to slide across an input image or feature map, performing a dot product at each position to extract features like edges or textures.
- This process generates feature maps, which highlight specific patterns and are used in tasks like image recognition and object detection.
- The 3x3 size is a balance between capturing contextual information and computational efficiency, as multiple 3x3 convolutions can form a larger effective receptive field with fewer parameters than a single larger kernel.

### Max Pooling Layer

- A max pooling layer reduces the spatial size of an image or feature map by dividing it into a grid of smaller regions and selecting the maximum value from each region to form a new, compressed representation.
- This downsampling process makes the network more efficient by decreasing the computational load and memory requirements, helps prevent overfitting, and provides a degree of translation invariance, allowing the network to recognize features regardless of their exact position.

**1. Performance Curves (`_curves.png`)**

This image contains two plots side-by-side:

*   **Left: ROC Curve (Receiver Operating Characteristic)**
    *   **X-axis (False Positive Rate):** The proportion of negative pixels (no fire) incorrectly classified as positive (fire).
    *   **Y-axis (True Positive Rate / Recall):** The proportion of actual fire pixels correctly identified by the model.
    *   **Interpretation:** A perfect model would go straight up to the top-left corner (0,1). The diagonal line represents a random guess. The **AUC (Area Under Curve)** score summarizes this; 1.0 is perfect, 0.5 is random. This tells you how well the model can distinguish between fire and non-fire pixels across *all possible thresholds*.

*   **Right: Precision-Recall Curve**
    *   **X-axis (Recall):** How many of the *actual* fires did we find? (Same as True Positive Rate).
    *   **Y-axis (Precision):** Of all the fires we predicted, how many were *actually* fires?
    *   **Interpretation:** This is crucial for imbalanced datasets like forest fires (where most pixels are *not* fire). A high curve towards the top-right corner is better. It shows the trade-off: to catch more fires (high recall), you usually have to accept more false alarms (lower precision).

**2. Visual Sample (`_visual.png`)**

This image shows a qualitative "sanity check" of the model's performance on a single example:

*   **Left: Input (Channel 0):** This visualizes one of the input features fed into the model (usually temperature or a moisture index). It gives context to the scene.
*   **Middle: Target Fire:** This is the "Ground Truth". The yellow/orange areas are where the fire *actually* occurred (according to the satellite data).
*   **Right: Prediction:** This is what the model *predicted*. Brighter/hotter colors indicate a higher probability of fire.
    *   **Interpretation:** Compare the Middle and Right images. Does the model's prediction shape match the actual fire? Is it missing areas (false negatives) or hallucinating fire where there is none (false positives)?
