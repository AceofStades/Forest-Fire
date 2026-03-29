# Forest-Fire Prediction and Simulation Platform: Comprehensive Technical Architecture

## 1. Introduction and Project Evolution
The **Forest-Fire** prediction project is an advanced, full-stack monorepo application aiming to predict, simulate, and mitigate the spread of wildfires using deep learning, satellite data, and dynamic pathfinding algorithms. 

The project is structured into three primary domains:
- **`Frontend/`**: A Next.js 15 application utilizing React 19, TypeScript, TailwindCSS, Shadcn UI, and Leaflet for dynamic mapping and interactive ML insights.
- **`Server/`**: A FastAPI Python backend serving machine learning predictions, probability grids, real-time PyTorch tensor advection, and dynamic D* Lite routing.
- **`Model/`**: The PyTorch-based Deep Learning research directory, handling dataset preprocessing (NetCDF), training pipelines, and the core U-Net + CBAM neural network architectures.

---

## 2. Dataset Engineering and Preprocessing
The model's ability to predict fire spread relies entirely on the quality and alignment of its multi-modal data. The raw data sources undergo rigorous preprocessing via `Model/preprocessing/merge_dynamic.py` and dynamic dataset loaders.

### 2.1 Raw Data Sources
1. **MODIS (Historical Fire Data):** The ground truth. Sourced from NASA's FIRMS dataset.
2. **ERA5-Land (Meteorological Data):** Provides hourly dynamic weather variables (temperature, dewpoint, soil moisture, evaporation, 10m wind components `u10`/`v10`, precipitation).
3. **DEM (Digital Elevation Model):** Topography is critical as fires travel exponentially faster uphill due to convective heat transfer.
4. **LULC (Land Use Land Cover):** Categorical data indicating vegetation density and fuel availability. 

### 2.2 Preprocessing Pipeline (`merge_dynamic.py`)
1. **Spatial and Temporal Filtering:** The massive raw datasets are cropped to a localized bounding box in Northern India (Lat: 28.7° N to 31.49° N, Lon: 77.5° E to 81.08° E).
2. **Rasterization & Alignment:** Static GeoTIFF maps are projected and interpolated using `rioxarray` to perfectly match the grid layout of the ERA5 weather data.
3. **Burn Scar Engineering:** A dynamic `Burn_Scar` channel is calculated on the fly using a cumulative sum (`cumsum`) of the fire occurrences. If a pixel has burned in the past, its `Burn_Scar` is set to 1.

---

## 3. Deep Learning Architectures & The "Identity Trap"

### 3.1 The Failure of Pure ML (Google NDWS Approach)
Initially, we attempted to train models to predict $F(t+1)$ directly from $F(t)$ (an end-to-end spatiotemporal approach similar to Google's NDWS). This failed spectacularly due to the **"Identity Trap"** and the **"Strobe Light Effect"** (satellites only pass over a few times a day). Pure ML attempts to learn fluid dynamics from discontinuous "flashes" of fire. To minimize loss, the model safely guesses that the fire at Day 2 will look exactly like the fire at Day 1, ignoring wind physics entirely and just expanding radially.

### 3.2 Our Solution: The Custom Hybrid Model
Because satellite temporal resolution is too low to teach a pure neural network fluid dynamics, we restructured the architecture:
1. **Machine Learning (Fuel Mapping):** The PyTorch U-Net (with CBAM attention) was stripped of its temporal sequence. It is completely blind to the current fire state. It now acts as a pure **Static Burn Susceptibility Model**, forced to synthesize only the weather, wind, and landscape (13 channels) to generate a topographical probability map.
2. **Physics Engine (Cellular Automaton):** The temporal advection (movement over time) is handled by a hard-coded PyTorch tensor convolution engine (`Server/app/sandbox_engine.py`). It reads the underlying ML Susceptibility Map, checks the interactive wind speed/direction, and computes mathematically rigorous spread physics (e.g., wind vector dot-products, uphill slope bias).

---

## 4. Pathfinding and Real-World Map Integration

To make the predictions actionable, we integrated advanced mapping and dual-pathfinding systems into the `interactive-dashboard`.

### 4.1 OpenRouteService (ORS) - Primary Street Routing
For real-world utility, the frontend utilizes the **OpenRouteService API** to calculate driving evacuation routes.
- **Dynamic Avoidance:** When a fire spreads on the UI, the frontend clusters the active fire pixels into 5x5 bounding boxes. These are sent to the ORS API as `avoid_polygons`, forcing the routing engine to recalculate street-level paths that dynamically dodge the burning zones.
- **Road Snapping:** If a user clicks deep in an off-road forest to start a route, the API payload includes `radiuses: [-1, -1]` to force an infinite-radius search, snapping the Start/Goal pins to the nearest valid real-world road.
- **Topology Layers:** The UI includes an interactive Esri `World_Street_Map` layer so users can visually verify the ORS road-snapping behavior against actual topological features.

### 4.2 D* Lite - Grid-based Fallback (`d_star_lite.py`)
Because ORS relies on external APIs and strict road networks, we maintain a robust **D* Lite** implementation in the Python backend as a fallback engine.
- If ORS fails, or if a user needs to navigate across raw wilderness (off-road), the frontend requests a path from the FastAPI `/get-safe-path` endpoint.
- The Python server injects `1.0` (max danger) into the ML probability matrix wherever a fire is currently active.
- D* Lite (a dynamic, incremental heuristic search) quickly computes the shortest path backward from the Goal to the Start across the probability tensor, safely routing around the high-risk zones without needing to recalculate the entire map from scratch.

---

## 5. The ML Insights Sandbox
A core feature of the platform is the `/ml-insights` page, containing an Interactive Architecture Comparison Sandbox. This tool allows users to visually compare our Custom Hybrid Model against a mock of the failing Pure ML (NDWS) model.

**How it works:**
- **Procedural Topography:** The frontend generates an interactive pseudo-random elevation map (DEM). Users can click anywhere to ignite custom fire clusters.
- **Tensor Advection Backend:** The UI streams the grid state, wind vectors, and topography to the FastAPI `/sandbox-step` endpoint at 5 FPS.
- **PyTorch Physics (`sandbox_engine.py`):** The backend uses lightning-fast 2D convolutions (`F.conv2d`) to process the advection of the entire 120x120 grid simultaneously. It penalizes upwind spread, accelerates downwind spread exponentially (creating realistic cigar-shaped plumes), and biases spread up positive topological slopes.
