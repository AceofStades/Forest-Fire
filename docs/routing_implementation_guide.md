# Routing & Pathfinding Implementation Guide

This document provides a deep, technical breakdown of how the dual-engine routing architecture is implemented in the Forest-Fire project. It covers both the frontend OpenRouteService (ORS) vector routing and the backend PyTorch-integrated D* Lite grid routing, specifically detailing how the system actively avoids predicted and active fire pixels.

## 1. The Dual-Engine Architecture Overview
Natural disasters require robust, fail-safe navigation. If a civilian is on a mapped road, they need street-level turn-by-turn directions. If they are deep in an unmapped forest, or if the external API goes offline/rejects the massive scale of the fire, the system must instantly fall back to a grid-based offline pathfinder. 

Our system achieves this via a seamless handoff:
1. **Primary:** OpenRouteService (Frontend / API)
2. **Fallback:** Custom D* Lite (Backend / Python)

---

## 2. Primary Engine: OpenRouteService (ORS)

The frontend (`Frontend/app/interactive-dashboard/MapSimulation.tsx`) attempts to use the ORS driving-car API first. However, passing thousands of individual burning pixels to a public API will instantly result in rate limits or "Polygon Area Exceeded" errors. 

### The Spatial Clustering Algorithm
To solve the API constraints, we dynamically group the 9x9m burning pixels into larger, optimized bounding boxes before sending them to ORS.

1. **Relevance Filtering:** 
   The system first draws a generous margin (approx. 40 grid units / ~30km) around the direct line between the Start and Goal points. Any active fires outside this zone are completely ignored. This prevents massive fires on the opposite side of the state from crashing the local routing request.
2. **Dynamic Block Clustering:**
   The remaining relevant fires are grouped into blocks. The algorithm starts with a tight `blockSize` of 2x2 grid units. It groups contiguous fires into bounding boxes. If there are still more than 20 boxes (the safe ORS limit), it dynamically scales the `blockSize` up (up to a cap of 6) to merge nearby fires into larger, shared bounding boxes.
3. **Midpoint Proximity Sorting:**
   If the algorithm hits the block size cap and still has $>20$ boxes, it calculates the physical distance of each box to the exact midpoint of the Start and Goal route. It then slices off the 20 boxes closest to the route, ensuring the most dangerous and relevant obstacles are prioritized.
4. **Padding & GeoJSON Compliance:**
   - **Padding:** Each bounding box is expanded outward by 1 pixel (~1km) to guarantee the ORS routing engine gives the fire a wide, safe berth.
   - **Counter-Clockwise Orientation (RFC 7946):** The ORS GeoJSON parser strictly requires the exterior ring of a polygon to be drawn counter-clockwise (NW $\rightarrow$ SW $\rightarrow$ SE $\rightarrow$ NE $\rightarrow$ NW). If drawn clockwise, ORS interprets it as an "Inverted Earth" polygon (avoid the whole planet *except* this box) and crashes.

### Road Snapping (`radiuses`)
If a user clicks deep in a forest (off the road network), standard APIs fail with "Unroutable Point." We bypass this by passing `radiuses: [-1, -1]` in the ORS payload, forcing an infinite-radius spatial search to snap the user's pin to the nearest valid highway.

---

## 3. Fallback Engine: Backend D* Lite

If ORS fails (due to missing API keys, massive fire areas, or network drops), the frontend silently catches the error and issues a `POST` request to the backend `/get-safe-path` endpoint.

### Why D* Lite?
Unlike A* (which searches from Start to Goal and must completely discard its memory and recalculate the entire map if an obstacle appears on its path), **D* Lite searches backwards from the Goal to the Start.**
When the cellular automata engine advances the fire by one step, D* Lite only updates the localized edge costs of the affected nodes. This incremental update allows for micro-second route recalculations, essential for a simulation running at 5 FPS.

### The Implementation (`Server/app/d_star_lite.py`)
The Python backend initializes the `DStarLite` class over the 320x400 Machine Learning probability matrix. The core logic dictating *how* the algorithm avoids fire is entirely handled in the **Cost Function Matrix**.

#### 1. 8-Way Connectivity
The algorithm evaluates the 8 neighboring cells around the current node. Diagonal movement is permitted to create natural, smooth paths rather than jagged stair-steps.

#### 2. The Heuristic Cost Function (`update_vertex`)
The `rhs` (one-step lookahead cost) evaluates the "danger" of stepping into a neighbor pixel based on the ML prediction grid:

```python
prob = self.grid[v[0], v[1]]
if prob >= 1.0:
    cost = float("inf")  # Active fire is impassable
else:
    cost = 1 + (prob * 1000 if prob > 0.6 else prob * 10)
```

*   **Base Distance Cost:** Moving to any neighbor costs `1`.
*   **The Impassable Wall (`prob >= 1.0`):** If a cell is actively burning (or flagged as an active fire by the frontend), its cost becomes infinity. The pathfinder physically cannot route through it.
*   **The Extreme Danger Zone (`prob > 0.6`):** If the ML model predicts a $>60\%$ chance of fire, the cost is multiplied by `1000`. This massive heuristic penalty forces the priority queue to aggressively warp the path *away* from advancing fire fronts long before the fire physically arrives.
*   **The Caution Zone (`prob <= 0.6`):** For low/moderate probabilities, a smaller `prob * 10` multiplier is applied. This encourages the algorithm to stay in "cool" areas (valleys, unburnable terrain) without taking massive, unnecessary detours.

### Path Extraction
Once the shortest path is verified (`compute_shortest_path`), the backend traces the gradient descent from the Start node down to the Goal, compiling an array of `[row, col]` tuples. This array is returned to the frontend, converted to Lat/Lng coordinates, and rendered as a green dashed Leaflet `Polyline`.
