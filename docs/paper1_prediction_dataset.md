# Research Paper 1: Multi-Modal Wildfire Prediction and Dataset Engineering

**Focus:** Dataset creation, preprocessing, overcoming ML traps (Identity Trap), Model Architectures (U-Net, ConvLSTM, Hybrid), and the NDWS Comparison.

## Abstract
Wildfire prediction models often struggle with spatial and temporal sparsity of ground truth data, leading to models that either oversimplify fire behavior or fall victim to predictive traps. This paper presents a novel multi-modal dataset engineered from satellite thermal anomalies, meteorological variables, and topographical maps across Uttarakhand, India. We detail the preprocessing pipelines necessary to align disparate spatiotemporal resolutions and the rigorous feature engineering applied to encode cumulative burn history and domain-specific thresholds. Furthermore, we explore multiple deep learning architectures—ranging from static spatial U-Nets to spatiotemporal ConvLSTMs. Crucially, we identify and dissect the "Identity Trap" (Persistence Bias) inherent in sequential predicting, where models achieve artificially high accuracy by merely copying the previous state ($F(t+1) = F(t)$) while completely ignoring environmental physics. We propose a paradigm shift from predicting explicit temporal frames to predicting a static "Burn Susceptibility Map," validated against baseline approaches such as Google's Next Day Wildfire Spread (NDWS).

## 1. Introduction
*   Contextualize the escalating threat of wildfires globally, focusing on the unique topographical challenges in Uttarakhand, India.
*   Discuss the critical gap in current predictive systems: traditional systems rely on broad meteorological indices (e.g., FWI) which lack pixel-level precision, while modern ML approaches often fail to generalize physical laws.
*   State the core objective: Constructing a high-resolution, multi-modal feature stack that integrates weather, topography, and active fire data to train robust spatial deep learning models.
*   Outline the paper's structure: Data engineering, architectural explorations, identifying the Identity Trap, and comparative evaluation.

## 2. Dataset Collection & Sources
*   **MODIS (Ground Truth):** Detail the use of NASA FIRMS thermal anomaly data (2016-present). Discuss the fundamental challenges: 1km spatial resolution and high temporal sparsity (satellites pass over a specific area only 2-4 times a day).
*   **ERA5-Land (Meteorological):** Explain the extraction of hourly climate variables (temperature `t2m`, dew point `d2m`, soil moisture `swvl1`, evaporation `e`, wind components `u10`, `v10`, total precipitation `tp`, low vegetation cover `cvl`).
*   **Topography (DEM):** Highlight the integration of Bhuvan/NRSC Digital Elevation Models. Emphasize why elevation, aspect, and slope are mathematically critical for fire spread (convective pre-heating pushes fire uphill).
*   **LULC & GHSL:** Detail the use of Land Use / Land Cover and Global Human Settlement Layers to identify burnable biomass vs. concrete/water barriers.

## 3. Data Engineering & Preprocessing
*   **Spatiotemporal Alignment:** Explain the complex pipeline required to interpolate raw GeoTIFFs (spatial) and NetCDF files (temporal) into a unified, strictly aligned 13-channel grid spanning a defined bounding box (Lat: 28.7–31.4, Lon: 77.5–81.0).
*   **Burn Scar Engineering (The Memory Problem):** Detail the algorithmic approach to tracking cumulative fire history (`cumsum`). Explain the physical constraint: a pixel that burns has its fuel depleted and cannot burn again immediately.
*   **Normalization & The "Blindness" Trap:** Discuss standard scaling vs. using 2nd/98th percentiles. Highlight a critical bug discovered: hardcoding sparse binary masks (like the fire label itself) through percentile normalization destroys the data. Explain the implementation of targeted bypasses to prevent "Normalization Blindness."
*   **Tensor Assembly:** Describe how these disparate variables are combined into PyTorch `.pt` tensors suitable for dataloading.

## 4. Deep Learning Architectures
*   **U-Net (Spatial Baseline):** Detail the implementation of a 2D U-Net designed to predict static Burn Susceptibility based purely on the 13 environmental features. Explain why hiding the "current fire state" forces the model to learn underlying environmental relationships.
*   **ConvLSTM / Hybrid (Spatiotemporal):** Discuss the attempt to ingest time-series sequences ($T-3, T-2, T-1$) to explicitly predict the frame at $T$. 
*   **Resource Constraints:** Elaborate on the massive VRAM requirements for 3D tensors and the necessary batch-size reduction and gradient accumulation strategies employed.

## 5. Major Training Challenges & The Identity Trap
*   **The "Strobe Light Effect":** Explain the difficulty of training on intermittent satellite data. Interpolating a 5-minute fire event across a 24-hour block creates static, unchanging tensors that confuse the model.
*   **The Identity Trap (Persistence Bias):** Deep dive into the most critical finding. Explain how, when tasked with predicting $F(t+1)$ given $F(t)$, models (like NDWS) achieve 99% accuracy simply by copying the input to the output. Prove mathematically and visually how this causes the model to entirely ignore wind, slope, and weather.
*   **Class Imbalance:** Detail how active fires represent <0.1% of the total pixels. Explain why standard Binary Cross-Entropy (BCE) loss completely failed (the model just predicts "0" everywhere).
*   **Algorithmic Solutions:** 
    *   **Architecture Shift:** Moving away from temporal frame prediction to static Susceptibility mapping.
    *   **Combined Loss Function:** Detail the mathematical implementation of Focal Loss ($\alpha=0.95, \gamma=2.0$) combined with Dice Loss to aggressively penalize missed fires.
    *   **Weighted Random Sampling:** Explain the custom Dataloader logic forcing the model to sample fire-active frames at a heavy 50:1 ratio compared to empty frames.

## 6. Results & NDWS Comparison
*   **Quantitative Metrics:** Present detailed tables showing F1-Score, Precision, and Recall comparing the Spatial U-Net against the Spatiotemporal ConvLSTM.
*   **The NDWS Comparison:** Rigorously contrast our static "Fuel Map" approach against the conceptual methodology of Google's Next Day Wildfire Spread (NDWS). 
*   **Conclusion on Pure ML:** Conclude why end-to-end Pure Machine Learning fundamentally fails at simulating complex physical advection without an explicitly defined physics engine (leading directly into Paper 2).

---

## 📸 Recommended Images / Figures to Create for this Paper:
1.  **The Multi-Modal Feature Stack:** A 3D isometric diagram showing the separate layers (Weather, DEM, LULC, MODIS) visually stacking into a single multi-channel PyTorch Tensor.
2.  **The Identity Trap Graph:** A dual-axis line chart showing Training and Validation F1 scores artificially spiking to 0.99 within 2 epochs, juxtaposed with side-by-side visual proof that the model output is just a pixel-perfect copy of the previous day's fire.
3.  **U-Net Architecture Diagram:** A standard encoder/decoder flowchart, specifically highlighting the integration of CBAM (Convolutional Block Attention Module) blocks and detailing the 13-channel input matrix.
4.  **Loss Function Heatmap:** A visual comparison of three map predictions: Ground Truth vs. Standard BCE Loss (mostly empty/black) vs. Focal+Dice Loss (tightly tracking the complex fire boundaries).
5.  **Results Table:** A comprehensive grid comparing Precision, Recall, F1, and Inference Time across the tested models and the theoretical NDWS baseline.
