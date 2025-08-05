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
