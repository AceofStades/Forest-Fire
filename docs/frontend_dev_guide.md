# Frontend Developer Guide: UI, Mapping, and State

This guide explains the architecture of the Next.js 15 frontend, the map rendering stack, and how the interactive sandboxes interface with the backend.

## 1. Tech Stack
- **Framework:** Next.js 15 (App Router)
- **UI Library:** React 19, TailwindCSS, Shadcn UI
- **Icons:** `lucide-react`
- **Mapping:** `leaflet`, `react-leaflet`, OpenRouteService (ORS)
- **Charts:** `recharts` (used in `/ml-insights`)

## 2. Interactive Map Dashboard (`/interactive-dashboard`)

The primary component is `MapSimulation.tsx`. It relies on several overlapping layers:

### A. The Basemap
We use `react-leaflet` to render a standard tile layer. Users can toggle between standard maps and Esri's `World_Street_Map` to visually verify road networks for the routing engine.

### B. PyTorch ML Canvas Overlay
The raw output from the U-Net model is a `320x400` numerical matrix representing spread probabilities.
- Instead of converting this massive matrix into 128,000 individual DOM `div` elements or SVG rectangles (which would crash the browser), we use a custom `<ImageOverlay>` or native HTML5 `<canvas>`.
- The numerical array is converted into raw pixel data (`ImageData`) using a color scale (e.g., Red for high probability, transparent for low probability) and painted directly onto the Leaflet map bounds: `[[28.718, 77.509], [31.490, 81.081]]`.

### C. The OpenRouteService (ORS) Integration
When a user clicks "Start" and "Goal" points, the UI attempts to find a valid street route using the ORS API.
- **Fire Clustering:** ORS API limits the complexity of `avoid_polygons`. If we send every individual burning pixel, the API fails. The frontend contains an algorithm that groups nearby burning pixels into larger 5x5 bounding boxes.
- **Payload:** These cluster boxes are sent to ORS. We inject `radiuses: [-1, -1]` to force the algorithm to snap the user's coordinates to the nearest physical road, preventing "Unroutable point" errors when clicking deep in the forest.

### D. The D* Lite Fallback
If ORS fails or returns no route, the frontend automatically falls back to the local FastAPI server. It sends the active fire pixels to `http://127.0.0.1:8000/get-safe-path`. The server calculates a grid-based D* Lite path across the wilderness and returns a raw coordinate array, which the frontend renders as a dotted `Polyline`.

## 3. The ML Insights Sandbox (`/ml-insights`)

The `ComparisonSandbox.tsx` component is an interactive educational tool demonstrating why the Custom Hybrid Model is superior to pure ML models (NDWS).

- **UI State:** It manages sliders for Wind Speed, Wind Direction, Temperature, and Slope Impact.
- **Terrain Generation:** On mount, it procedurally generates a pseudo-random topographical elevation map (DEM) using simple cellular smoothing, mapping the values to green/tan colors.
- **The Loop:** When the user clicks "Play", the frontend enters a `useEffect` loop, running at 5 FPS. It packages the raw 120x120 byte arrays (`ndwsGrid`, `customGrid`, `terrainGrid`) alongside the slider parameters and POSTs them to the backend `/sandbox-step`.
- **Rendering:** It receives the newly calculated arrays from the PyTorch physics engine and uses native Canvas API `ctx.putImageData()` to instantly repaint the 14,400 pixels without React re-render lag.
