# Dataset Comparison: Custom Multi-Modal Dataset vs. Google NDWS Dataset

This document provides an extremely detailed breakdown of the dataset engineering behind our Forest-Fire predictive model compared to the dataset methodology used by Google's Next Day Wildfire Spread (NDWS) project. 

## 1. Introduction to the Data Problem
Wildfire prediction is inherently starved for high-quality, continuous data. Satellites orbiting the Earth do not provide a 24/7 continuous video feed of a fire; they provide "strobe light" snapshots (passes) a few times a day. How a dataset handles this sparsity, aligns meteorological data, and incorporates terrain dictates the absolute ceiling of the model's physical accuracy.

---

## 2. Google NDWS Dataset Methodology
Google's NDWS dataset is a benchmark in the field, but it was built with specific constraints and targets in mind (primarily massive, multi-day fires in the United States, such as California).

*   **Satellite Source:** NDWS heavily relies on GOES (Geostationary Operational Environmental Satellites). GOES provides excellent temporal resolution (snapshots every 5-15 minutes) but suffers from coarse spatial resolution (often 2km to 4km per pixel). 
*   **Target Temporal Window:** It is designed to predict the *next day's* spread. It maps the fire perimeter at Day $T$, inputs weather conditions for Day $T+1$, and attempts to predict the perimeter at Day $T+1$.
*   **Input Features:** 
    *   Fire state at Day $T$.
    *   Elevation (DEM).
    *   Meteorological data (Wind speed, Wind direction, Temperature, Humidity).
    *   Vegetation indices.
*   **The NDWS Flaw (Sparsity & Scale):** Because NDWS targets massive fires over 24-hour windows, the dataset inherently glosses over micro-topography. In rugged, vertical terrains (like the Himalayas), a 2km-4km pixel completely blurs out a massive valley and a mountain peak into a single flat average, destroying the physical mechanics of slope-driven convective heating.

---

## 3. Our Custom Dataset Methodology (The Uttarakhand Focus)
Our dataset was explicitly engineered to handle hyper-rugged terrain (the Himalayas in Uttarakhand, India: Lat 28.7–31.4, Lon 77.5–81.0) and to decouple the "current fire" from the "environmental fuel."

### A. High-Resolution Spatial Anchoring
Instead of geostationary satellites, we utilize polar-orbiting satellites combined with hyper-local terrain maps.
*   **MODIS (FIRMS) Thermal Anomalies:** We use MODIS data which provides a much tighter 1km spatial resolution. While temporal resolution is lower (2-4 passes a day), the spatial accuracy allows us to pinpoint exactly *which side* of a mountain ridge is burning.
*   **Bhuvan/NRSC Topography (DEM):** Digital Elevation Models are treated as a first-class feature. Wildfires physically move much faster uphill due to convective pre-heating of fuels above the flame. Our grid captures high-frequency elevation changes that coarse global datasets miss.

### B. The 13-Channel Multi-Modal Stack
We engineered a unified spatial grid where every single pixel contains 13 perfectly aligned temporal and spatial features.
1.  **ERA5-Land (Hourly Meteorology):** `t2m` (Temperature), `d2m` (Dew Point), `swvl1` (Soil Moisture), `e` (Evaporation), `u10` & `v10` (U and V Wind Vectors), `tp` (Total Precipitation), `cvl` (Low Vegetation Cover).
2.  **Topography:** Elevation, localized Slope (derived gradient), Aspect (facing direction of the slope).
3.  **LULC & GHSL:** Land Use and Global Human Settlement Layers (to act as unburnable barriers like concrete or lakes).

### C. Advanced Data Engineering Techniques
Where our dataset drastically diverges from standard NDWS preprocessing is how we mathematically handle the temporal history and normalization.

*   **Burn Scar Memory (`cumsum` tracking):** A forest cannot burn twice in the same week. NDWS models often struggle with fires turning around and burning backward over their own ashes. We engineered a cumulative sum tracker: when a pixel registers a fire, its "Fuel Availability" channel is permanently depleted in the tensor for the remainder of the temporal sequence.
*   **Avoiding "Normalization Blindness":** Standard machine learning pipelines apply a Standard Scaler (Mean 0, Std 1) or Min-Max Scaler to all input channels. **This destroys sparse binary masks.** Because fires represent $<0.1\%$ of the map, normalizing the fire channel squashes the $1.0$ (fire) values into an indistinguishable decimal near $0.0$, making the model "blind." We wrote custom preprocessing logic that uses 2nd/98th percentiles for weather data, but strictly hard-bypasses normalization for categorical/binary masks (Fire, Burn Scars, LULC).

---

## 4. Summary of Dataset Differences

| Feature | Google NDWS Dataset | Our Custom Dataset |
| :--- | :--- | :--- |
| **Primary Goal** | 24-Hour massive fire perimeter expansion. | Micro-scale susceptibility and physical fuel mapping. |
| **Target Geography** | US / California (broad, sweeping fires). | Uttarakhand, India (extreme vertical Himalayan topography). |
| **Spatial Resolution** | Coarse (2km - 4km). | High (1km exact gridding). |
| **Temporal Resolution**| High (15 min GOES). | Moderate (Hourly ERA5, Sparse MODIS). |
| **Normalization** | Standard regional scaling. | Custom percentile bypass for sparse binary masks to prevent blindness. |
| **Fuel Memory** | Often implicit via vegetation indices. | Explicitly tracked via engineered `cumsum` burn-scar matrices. |
