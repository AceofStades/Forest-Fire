# Forest-Fire Prediction and Simulation Platform: Comprehensive Technical Architecture

## 1. Introduction and Project Evolution
The **Forest-Fire** prediction project is an advanced, full-stack application aiming to predict, simulate, and mitigate the spread of wildfires using deep learning, satellite data, and dynamic pathfinding algorithms. 

The project has evolved significantly over its lifecycle:
- **Phase 1 (Legacy Implementation - Feb 2025):** The original models focused on relatively static datasets. We implemented a basic `UNet`, an `LSTM`, a `ConvLSTM`, and a `HybridFireNet` (combining a spatial UNet encoder with a temporal ConvLSTM bottleneck). The data pipeline was static, often evaluating single frames without a continuous spatiotemporal flow.
- **Phase 2 (Dynamic Spatiotemporal Pipeline - Current):** To predict actual *spread*, we transitioned to a fully dynamic, hourly 4D dataset. We ingested satellite fire spots, hourly meteorological data, and static topological maps, merging them into a unified `.nc` (NetCDF) feature stack. The architectures were refined down to two core, highly optimized implementations: a spatial **UNet** and a spatiotemporal **ConvLSTM**.
- **Phase 3 (Integration & Unified Dashboard):** The backend was built using FastAPI to serve model inferences to a modern Next.js 15 React application. We consolidated fragmented simulation pages into a single, unified `interactive-dashboard`. Crucially, a **D* Lite** pathfinding algorithm was integrated to calculate safe evacuation routes dynamically based on predicted fire probabilities and real-time interactive simulation states.

---

## 2. Dataset Engineering and Preprocessing
The model's ability to predict fire spread relies entirely on the quality and alignment of its multi-modal data. The raw data sources undergo rigorous preprocessing via `Model/preprocessing/merge_dynamic.py` and dynamic dataset loaders.

### 2.1 Raw Data Sources
1. **MODIS (Historical Fire Data):** The ground truth. Sourced from NASA's FIRMS dataset. The original dataset (`modis_2016_India.csv`) contained ~88,000 thermal anomaly readings across India for the entire year of 2016.
2. **ERA5-Land (Meteorological Data):** Provides hourly dynamic weather variables at a ~9km resolution, including 2m temperature (`t2m`), 2m dewpoint (`d2m`), soil moisture (`swvl1`), evaporation (`e`), 10m wind components (`u10`, `v10`), and precipitation (`tp`).
3. **DEM (Digital Elevation Model):** Sourced from Bhuvan/NRSC. Topography is critical as fires travel exponentially faster uphill due to convective heat transfer and wind dynamics.
4. **LULC (Land Use Land Cover):** Categorical data indicating vegetation density and fuel availability. 
5. **GHS_BUILT (Global Human Settlement Layer):** Indicates built-up urban infrastructure, acting as potential ignition sources or fuel breaks.

### 2.2 Preprocessing Pipeline (`merge_dynamic.py`)
1. **Spatial and Temporal Filtering:** The massive raw datasets were cropped to a localized bounding box in Northern India (Lat: 28.7° N to 31.3° N, Lon: 77.5° E to 80.5° E) and isolated to the critical pre-monsoon fire season (April 1 to May 31, 2016). This reduced the MODIS dataset to ~2,800 highly relevant fire spots.
2. **Rasterization & Alignment:** The static GeoTIFF maps (DEM, LULC, GHS) were projected and interpolated using `rioxarray` to perfectly match the grid layout of the ERA5 weather data. The discrete MODIS point coordinates were mapped to the nearest hourly time-slot and rasterized into a 3D binary array (Time, Latitude, Longitude).
3. **Burn Scar Engineering:** Fire destroys its own fuel. In `src/dataset.py`, a dynamic `Burn_Scar` channel is calculated on the fly using a cumulative sum (`cumsum`) of the fire occurrences. If a pixel has burned in the past, its `Burn_Scar` is set to 1, teaching the model that fire cannot return to that exact pixel immediately.

---

## 3. Deep Learning Architectures
The system frames wildfire prediction as an image segmentation problem.

### 3.1 UNet (Spatial Prediction)
The UNet is a fully convolutional network that analyzes a single time frame. 
- **Input:** 13 channels (Weather, Elevation, LULC, GHS, Burn_Scar). It is *blind* to the current fire state.
- **Optimization:** Early versions of the model suffered from "persistence memorization"—it simply copied the input fire state to the output. To combat this, we removed the fire state from the input entirely. The model now acts as a pure **Static Burn Susceptibility (Fuel) Model**, forced to synthesize the weather and vegetation features to generate a topographical probability map. We also injected aggressive spatial `Dropout2d` layers (rates of 0.3 and 0.5) into the bottleneck.

### 3.2 ConvLSTMFireNet (Spatiotemporal Prediction)
Fires are a function of both space and time. 
- **Architecture:** The ConvLSTM ingests sequences of inputs (e.g., $T-3, T-2, T-1, T$) via stacked Convolutional LSTM cells. It maintains a hidden state matrix that captures momentum.
- **Optimization:** Spatiotemporal sequences demand massive VRAM. To resolve Out-of-Memory (OOM) GPU errors, we engineered the training loop to dynamically reduce the physical `batch_size` to 2 for ConvLSTM, while quadrupling the `ACCUMULATION_STEPS`. 

---

## 4. Addressing Critical Data Challenges

Developing this system uncovered several severe challenges regarding geospatial machine learning, ultimately leading to a Hybrid ML + Math architecture.

### 4.1 The "Strobe Light" Effect and the Identity Trap
**The Problem:** The initial dynamic dataset produced an F1 score of exactly `0.0`. Because MODIS is a satellite in low Earth orbit, it only takes snapshots a few times a day. In hourly data, a fire would appear at 10:00 AM, the map would be completely blank at 11:00 AM, and a fire would reappear at 4:00 PM. 
**The Failed Solution:** We initially applied a **24-hour persistence interpolation**. If a fire was detected at time $T$, we forward-filled that fire pixel for the next 24 hours. However, because $T$ and $T+1$ now overlapped by 99.9%, the model learned the "Identity Trap"—it simply copied the input to the output, achieving a 0.99 F1 score while fundamentally failing to learn any fluid dynamics or physics.

### 4.2 The Hybrid Pivot: ML for Fuel + CA for Spread
**The Solution:** Because the temporal resolution of satellites is fundamentally too low to teach a pure neural network fluid dynamics, we restructured the architecture:
1. **Machine Learning:** The PyTorch UNet was stripped of its temporal sequence. It now predicts a static **Burn Susceptibility Map** based only on the weather and landscape.
2. **Physics Engine:** The temporal advection (movement over time) was moved into a hard-coded **Cellular Automaton (CA)** in the Next.js React frontend. The CA looks at the underlying ML Susceptibility Map, checks the user's interactive Wind Speed and Direction sliders, and pushes the fire mathematically **day-by-day** across the valid fuel paths.

### 4.3 The Normalization Blindness Bug
**The Problem:** The dataset automatically normalized all inputs to a `[0, 1]` range based on the 2nd and 98th percentiles to handle outliers. However, because fires are extremely rare anomalies (< 0.1% of the map), the 98th percentile for `Burn_Scar` was `0.0`. Therefore, the max value was saved as 0. When normalizing, the code inadvertently erased all actual historical scars.
**The Solution:** We explicitly hardcoded sparse, binary layers to bypass percentile scaling and utilize a hard maximum of `1.0`.

---

## 5. Loss Functions and Imbalance Handling
Wildfires are a severe class imbalance problem. Over 95% of the map is just empty land. A model trained on standard Binary Cross-Entropy (BCE) will instantly learn to predict "No Fire" everywhere to achieve 95% accuracy.

1. **Combined Loss (Focal + Dice):** This is our primary, highly optimized loss function.
   - **Focal Loss ($\alpha=0.95, \gamma=2.0$):** Dynamically scales the gradient. It practically zeros out the loss from easy, obvious "empty" pixels and forces the neural network's backpropagation to focus almost exclusively on the edges of the fire.
   - **Dice Loss:** A region-based loss that calculates the Intersection over Union (IoU) of the predicted fire versus the actual fire, forcing the model to care about the structural shape of the spread.
2. **Weighted Random Sampling:** We implemented a custom DataLoader sampler that oversamples frames containing fire targets by a ratio of **50:1**. This ensures every training batch contains actionable fire dynamics, preventing the model from stagnating during long periods of clear weather.

---

## 6. Backend Integration and Simulation
The machine learning models are not isolated; they are integrated into a robust real-world backend.

### 6.1 FastAPI Server and Lifespan State
The backend `Server/app/main.py` uses FastAPI. To ensure fast inference, the trained PyTorch model weights and global dataset statistics (`stats_cache_fi.pkl`) are loaded into RAM once during the application's startup lifecycle using FastAPI's `@asynccontextmanager lifespan` hook. The server actively maps inference variables against the `DYNAMIC_interpolated.nc` NetCDF file to serve localized predictions based on specific lat/lon boundaries.

### 6.2 D* Lite Dynamic Pathfinding (`d_star_lite.py`)
To make the predictions actionable, we integrated the **D* Lite** algorithm. When the ML model generates a probability matrix of a fire spreading, this matrix is fed directly into the D* Lite environment. 
- D* Lite searches backward from the user's Goal to their Start position.
- The cost function heavily penalizes nodes with high fire probability.
- Because D* Lite is dynamic, if the UNet model predicts a sudden wind shift that blocks the path in real-time, D* Lite can quickly recompute a new, safe evacuation route without having to recalculate the entire map from scratch.

### 6.3 Interactive React-Leaflet Simulation Sandbox
The frontend (Next.js 15) utilizes `react-leaflet` and HTML5 Canvas layers to dynamically overlay the 320x400 output matrices from the PyTorch backend onto a real-world map of Uttarakhand (`Lat: 28.718 to 31.491`, `Lon: 77.509 to 81.082`). 
- **Historical Analysis:** Users can select historical events from April/May 2016. The backend pulls the actual `MODIS_FIRE_T1` tensor sequence to show a side-by-side animated CA playback of the *Predicted Spread* vs the *Actual Ground Truth Spread* over a 48-hour period.
- **Sandbox Mode:** Government officials can click dynamically on the map to ignite virtual fires, adjusting real-time wind speeds and vectors on the UI. The Cellular Automata leverages the UNet's probability logic and immediately biases the spread to reflect wind constraints over the real topography.
