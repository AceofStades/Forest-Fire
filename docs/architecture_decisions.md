# Architecture & Design Decisions

This document tracks the crucial structural and architectural decisions made across the Forest-Fire monorepo.

## 1. The Machine Learning Pivot: Overcoming the "Identity Trap"

### The Core Issue: The "Strobe Light" Effect
The raw dataset relies on MODIS satellite data, which only provides a snapshot of the fire 1-2 times per day. In our hourly dataset array, this meant the fire was entirely static for 23 hours a day, jumping abruptly once the satellite passed over.
*   **The Initial Failure:** When we asked the U-Net to predict the fire state at T+1 while passing the current fire map (`MODIS_FIRE_T1`) as an input channel, the model learned an "Identity Function." Because T and T+1 overlap by >99.9%, it simply copied the input fire channel directly to the output. It ignored wind, slope, and temperature entirely, instantly achieving a fake 0.99 F1 score while fundamentally failing to learn any physics.

### The Solution: ML for Fuel + Tensor CA for Spread
Because the temporal resolution of satellites is fundamentally too low to teach a pure neural network fluid dynamics hour-by-hour, we restructured the architecture:
1.  **Strip the Temporal Cheat:** The UNet is **no longer a temporal predictor**. It is now a **Static Burn Susceptibility (Fuel) Model**. It receives 13 channels (Weather, Vegetation, Topography, Burn Scar) but is completely blind to where the fire currently is. It outputs a 2D probability matrix representing inherent flammability.
2.  **PyTorch Tensor Physics Engine:** Instead of training ML to guess wind vectors, we explicitly enforce them. The temporal advection (movement over time) runs in the Python backend via a **Cellular Automaton (CA)** powered by PyTorch 2D Convolutions (`torch.nn.functional.conv2d`). This allows us to process 14,400 cells simultaneously, applying mathematically rigorous penalties for upwind spread and bonuses for uphill slope spread.

## 2. Frontend & Map Infrastructure

### Next.js 15 App Router
The frontend transitioned from standalone React scripts to the modern **Next.js 15 App Router**. This provides built-in server-side rendering (SSR), optimized routing, and a clean monorepo separation (`Frontend/`). The UI is built using **TailwindCSS** and **Shadcn UI** for strict, highly-polished aesthetics.

### React-Leaflet for Spatiotemporal Mapping
We selected **Leaflet** (via `react-leaflet`) over Google Maps/Mapbox for the Interactive Dashboard. Leaflet provides superior flexibility for injecting raw HTML5 Canvas layers (necessary for rendering the 320x400 ML probability matrices) and custom GeoJSON vector polygons directly on top of Esri `World_Street_Map` tiles without API cost constraints.

## 3. Dual-Pathfinding Architecture

To make the fire spread actionable, the platform requires an evacuation routing system. We implemented a hybrid dual-system approach:

### OpenRouteService (ORS) - The Primary Engine
For real-world utility, users need directions that follow actual streets and highways. 
*   **Decision:** We integrated the open-source ORS API. 
*   **Challenge:** ORS crashes if given complex, overlapping fire pixels or if the user clicks a coordinate slightly off a road.
*   **Solution:** We built a clustering algorithm in `MapSimulation.tsx` that groups individual fire pixels into 5x5 bounding boxes. We pass these clusters to ORS as `avoid_polygons`. To fix the off-road clicking, we inject `radiuses: [-1, -1]` into the payload, forcing ORS to infinitely snap the user's Start and Goal coordinates to the nearest valid road network.

### D* Lite - The Grid Fallback Engine
ORS requires an internet connection and strict road networks. In a deep forest fire, officials may need off-road evacuation metrics.
*   **Decision:** We implemented the **D* Lite** algorithm directly in the Python FastAPI backend (`Server/app/d_star_lite.py`).
*   **Why D* Lite?:** Unlike A* which must recalculate the entire map if a new fire spawns, D* Lite is dynamic. It searches backward from Goal to Start. When the CA engine spawns a new fire blocking the path, D* Lite only updates the affected node costs and instantly recalculates the new safest path across the raw ML probability tensor grid.
