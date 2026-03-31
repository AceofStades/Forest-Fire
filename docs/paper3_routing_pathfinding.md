# Research Paper 3: Dynamic Evacuation Routing in Evolving Disaster Scenarios

**Focus:** The integration of real-world street routing (OpenRouteService) and dynamic grid-based pathfinding (D* Lite) to navigate around actively spreading ML-simulated fires.

## 1. Introduction
*   The critical failure of standard navigation systems (Google Maps, Waze) during natural disasters: they route people *into* the hazard if the road isn't explicitly marked closed by authorities yet.
*   Objective: Building a routing engine that predicts where the fire *will be* and actively routes civilians and responders safely around the shifting perimeter.

## 2. The Dual-System Pathfinding Architecture
*   Why a single routing algorithm isn't enough. The necessity of a hybrid approach (Street-level API + Offline Grid fallback).

## 3. Primary Engine: OpenRouteService (ORS) Integration
*   **Vector Geometry & Constraints:** ORS allows passing `avoid_polygons` to dodge specific areas.
*   **The Polygon Clustering Algorithm:** Passing 10,000 individual burning 9x9m pixels to an API will crash it. We implemented a spatial clustering algorithm that groups nearby burning pixels into simplified `5x5` bounding boxes.
*   **Infinite Road-Snapping:** Addressing the "Unroutable Point" error. When a user clicks deep in a forest off the road network, we modify the payload to include `radiuses: [-1, -1]`, forcing an infinite-radius spatial search that snaps the Start/Goal pins to the nearest valid highway.

## 4. Fallback Engine: Dynamic D* Lite over ML Tensors
*   **When ORS Fails:** Deep wilderness navigation or API offline scenarios.
*   **Why D* Lite?** 
    *   Compare A* vs D* Lite. If a fire spawns and blocks a path, A* must recalculate the entire map from scratch. 
    *   D* Lite searches backward from the Goal to the Start. When the Cellular Automaton Engine advances the fire, D* Lite incrementally updates only the affected node costs, resulting in microsecond recalculations.
*   **The Cost Function Matrix:**
    *   The `DStarLite` algorithm runs directly on the ML output matrix.
    *   Cost = Distance Penalty + Extreme Penalty if `ML Probability > 0.6`.
    *   Injecting real-time satellite fire spots as `1.0` (Impassable) barriers.

## 5. Frontend UI/UX and Interactive Verification
*   How the path is rendered using React-Leaflet Polylines over Esri `World_Street_Map` tiles.
*   The live-updating badge UI that explicitly informs the user whether they are currently being routed by "ORS (Street Network)" or "D* Lite (Grid Fallback)".

---

## 📸 Recommended Images / Figures to Create for this Paper:
1.  **Dual-Routing Decision Tree:** Flowchart showing User Input $\rightarrow$ ORS Attempt $\rightarrow$ Success? (Render Route) $\rightarrow$ Fail? (Trigger D* Lite Python endpoint).
2.  **Fire Clustering Visual:** A screenshot of the map showing individual fire pixels, overlaid with the large transparent bounding boxes that are actually sent to the ORS API.
3.  **D* Lite Cost Map:** A visual representation of the grid where the path is shown physically bending and warping specifically to avoid an ML "Hotspot" (red zone) even if there is no actual fire there yet.
4.  **A* vs D* Lite Recalculation Graph:** A chart showing the compute time (ms) of A* vs D* Lite as the fire expands over 50 simulation steps. D* Lite should stay relatively flat (fast), while A* compute time spikes.
