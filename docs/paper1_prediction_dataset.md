# Research Paper 1: Multi-Modal Wildfire Prediction and Dataset Engineering

**Focus:** Dataset creation, preprocessing, overcoming ML traps (Identity Trap), Model Architectures (U-Net, ConvLSTM, Hybrid), and the NDWS Comparison.

## 1. Introduction
*   Context on wildfires in Uttarakhand/India.
*   The gap in current predictive models (relying purely on weather indices rather than pixel-level ML).
*   Objective: Building a high-resolution, multi-modal feature stack to train spatial deep learning models.

## 2. Dataset Collection & Sources
*   **MODIS (Ground Truth):** NASA FIRMS thermal anomalies (2016). Discuss spatial resolution and temporal sparsity (satellite passes).
*   **ERA5-Land (Meteorological):** Hourly variables (t2m, d2m, swvl1, e, u10, v10, tp, cvl).
*   **Topography (DEM):** Bhuvan/NRSC data. Crucial for understanding slope (fire moves faster uphill).
*   **LULC & GHSL:** Vegetation density and human settlement markers.

## 3. Data Engineering & Preprocessing
*   **Spatiotemporal Alignment:** Interpolating raw GeoTIFF and NetCDF files to a unified 13-channel grid over a specific bounding box (Lat: 28.7–31.4, Lon: 77.5–81.0).
*   **Burn Scar Engineering:** Algorithmically tracking cumulative fire history (`cumsum`). A pixel that burns has its fuel depleted.
*   **Normalization:** Standard scaling vs. 2nd/98th percentiles. Hardcoding sparse binary masks (like fires) to bypass percentiles to avoid "Normalization Blindness."

## 4. Deep Learning Architectures
*   **U-Net (Spatial):** Predicting static Burn Susceptibility based purely on 13 environmental features (blind to current fire state).
*   **ConvLSTM / Hybrid (Spatiotemporal):** Attempting to ingest Sequences ($T-3, T-2, T-1$) to predict $T$. Discuss the massive VRAM requirements and batch-size reduction strategies.

## 5. Major Training Challenges
*   **The "Strobe Light Effect":** Satellites only capture fires intermittently. Interpolating fires across 24 hours created static, unchanging blocks of data.
*   **The Identity Trap (Persistence Bias):** When tasked with predicting $F(t+1)$ given $F(t)$, models achieved 99% accuracy simply by copying the input to the output, entirely ignoring wind/weather.
*   **Class Imbalance:** Fires represent <0.1% of pixels. Standard BCE loss failed.
*   **Solutions:** 
    *   Shifted to predicting a *Static Susceptibility Map* (removing $F(t)$ from input).
    *   **Combined Loss Function:** Focal Loss ($\alpha=0.95, \gamma=2.0$) + Dice Loss.
    *   **Weighted Random Sampling:** Forcing dataloaders to sample fire-active frames at a 50:1 ratio.

## 6. Results & NDWS Comparison
*   **Quantitative Metrics:** Show F1, Precision, Recall for U-Net vs ConvLSTM.
*   **The NDWS Comparison:** Contrast our "Fuel Map" approach against Google's Next Day Wildfire Spread (NDWS) methodology. Explain why end-to-end Pure ML fails at physical advection without a physics engine.

---

## 📸 Recommended Images / Figures to Create for this Paper:
1.  **The Multi-Modal Feature Stack:** A 3D diagram showing layers of Weather, DEM, LULC, and MODIS stacking into a PyTorch Tensor.
2.  **The Identity Trap Graph:** A line chart showing Training/Validation F1 scores of the NDWS model artificially spiking to 0.99, juxtaposed with visual proof that it just "copies" the previous day's fire.
3.  **U-Net Architecture Diagram:** Standard encoder/decoder visual, specifically highlighting the CBAM attention blocks and 13-channel input.
4.  **Loss Function Heatmap:** A visual comparison of a prediction using standard BCE (mostly empty) vs. Focal+Dice (tightly tracking the fire boundaries).
5.  **Results Table:** Comparing Precision/Recall/F1 across the different models and the NDWS baseline.
