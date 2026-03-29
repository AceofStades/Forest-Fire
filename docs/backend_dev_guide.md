# Backend Developer Guide: API, Pathfinding, and Physics Engines

This guide explains how the FastAPI backend integrates the machine learning models (PyTorch), pathfinding algorithms (D* Lite), and the realtime physical advection engines for the interactive sandboxes.

## 1. Environment & Setup

The backend server relies entirely on `uv` for modern, blazing-fast Python environment management. 
- **Dependencies:** All dependencies are managed in the root `pyproject.toml` and `uv.lock`.
- **Run the Server:**
  ```bash
  cd Server
  uv run uvicorn app.main:app --reload --port 8000
  ```

## 2. Directory Structure

```bash
Server/
├── app/
│   ├── main.py             # FastAPI entrypoint and HTTP endpoints
│   ├── d_star_lite.py      # D* Lite algorithm implementation
│   ├── model_wrapper.py    # Utility for loading PyTorch ML models
│   ├── sandbox_engine.py   # PyTorch Tensor 2D Convolution physics engine
│   ├── schemas.py          # Pydantic validation schemas
│   └── models/             # Legacy directory for isolated weights
```

## 3. Core API Endpoints

### A. Real-world Pathfinding (`POST /get-safe-path`)
Used as a robust, offline-capable fallback when the primary OpenRouteService (ORS) street-routing API fails or the user is deep in off-road wilderness.
1. The frontend sends the `start` tuple, `goal` tuple, and an array of `active_fires`.
2. The backend fetches the pre-computed ML `fire_grid` probability matrix from memory.
3. It iterates through the `active_fires` array and artificially injects `1.0` (maximum danger) into the grid at those coordinates.
4. The `DStarLite` class computes the shortest path across the tensor from Goal to Start, safely skirting around the probability hotspots.

### B. Interactive Physics Sandbox (`POST /sandbox-step`)
Drives the side-by-side architecture comparison on the `/ml-insights` page. 
1. The frontend streams the 14,400 (120x120) state grid, topography (DEM), and user-selected wind/temperature parameters at 5 frames per second.
2. The endpoint routes the data to `sandbox_engine.py`.
3. **NDWS Mock (`run_ndws_step`):** Runs a naive radial expansion using standard convolution, simulating the "Identity Trap" where models ignore physics.
4. **Custom Hybrid (`run_custom_hybrid_step`):** Converts the grids to PyTorch Tensors (`torch.tensor`). It uses `torch.nn.functional.conv2d` across 8 directional kernels to simultaneously calculate spread probabilities for the entire map. It applies rigorous dot-product mathematics to penalize upwind spread and exponentially boost downwind/uphill spread based on the DEM tensor.

### C. Historical Event Data (`GET /event-data/{idx}`)
Fetches multi-modal NetCDF arrays to project onto the Leaflet map.
- Runs the `U-Net` model in real-time for a specific hourly index.
- Returns `probGrid` (static ML susceptibility), `groundTruth` (actual satellite fire flashes), and geographical bounding coordinates (`[Lat, Lon]`) to correctly overlay the 320x400 matrices onto Esri street maps in the UI.

## 4. State Management (Lifespan Hook)
To ensure the API responds instantly without loading gigabytes of weights on every request, the model (`best_unet.pth`) and required NetCDF dataset statistics (`stats_cache_fi.pkl`) are loaded into RAM exactly once during the server's startup phase using FastAPI's `@asynccontextmanager lifespan` hook in `main.py`. The `ml_artifacts` dictionary acts as a globally accessible state object for all active endpoints.
