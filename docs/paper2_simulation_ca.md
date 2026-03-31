# Research Paper 2: A Hybrid Digital Twin for Wildfire Simulation via Tensor-Based Cellular Automata

**Focus:** The Physics Engine, bridging the gap between static ML predictions and dynamic temporal spread, and the architecture of the Interactive Sandbox.

## 1. Introduction
*   The limitation of pure Deep Learning in simulating fluid dynamics over time (due to low temporal resolution of training data).
*   The concept of a Digital Twin: creating an interactive, real-time clone of a forest environment.
*   Objective: Fusing ML-generated "Fuel Maps" with explicit mathematical advection engines.

## 2. The Hybrid Architecture Pivot
*   Why we abandoned end-to-end spatiotemporal ML for active simulation.
*   **Component 1 (Machine Learning):** The PyTorch U-Net generates a localized, static probability matrix (Susceptibility based on weather/terrain).
*   **Component 2 (Physics Engine):** A Cellular Automaton (CA) that ingests the ML matrix and applies fluid advection.

## 3. Tensor-Based Cellular Automata (The Physics Engine)
*   **The Flaw of Standard CA:** Iterating pixel-by-pixel in JavaScript is slow and can create blocky, non-physical artifacts.
*   **The PyTorch Convolution Solution:** Using `torch.nn.functional.conv2d` across 8 directional kernels to process the entire 14,400-pixel grid simultaneously.
*   **Mathematical Advection Rules:**
    *   *Wind Vectors:* Calculating dot products between the spread vector and wind direction. Using exponential scaling to aggressively boost downwind spread while heavily penalizing upwind spread (creating authentic "cigar-shaped" plumes).
    *   *Topographical Slope:* Shifting the DEM tensor to calculate localized gradients. Accelerating uphill fire spread (convective heating) and slowing downhill spread.
    *   *Fuel Depletion:* Tracking state transitions ($0 = empty, 1 = burning, 2 = scarred/depleted$).

## 4. The Interactive Digital Twin Dashboard
*   **System Design:** Next.js 15 frontend streaming user parameters (Wind Speed, Direction, Temperature, Slope Impact) to a FastAPI backend at 5 FPS.
*   **Procedural Validation:** How the sandbox generates pseudo-random topographical maps to validate physical rules before deploying to real-world coordinates.
*   **Real-World Integration:** Overlaying the tensor calculations onto Esri Street Maps via Leaflet/HTML5 Canvas overlays.

## 5. Comparative Evaluation
*   **Pure ML (NDWS mock) vs. Hybrid CA:** Demonstrating how Pure ML expands radially (ignoring wind) because it relies on past states, whereas the Hybrid CA respects physical laws.

---

## 📸 Recommended Images / Figures to Create for this Paper:
1.  **System Flowchart:** showing the U-Net outputting a Probability Matrix $\rightarrow$ fed into the CA Engine $\rightarrow$ combined with User Sliders $\rightarrow$ outputting the next Fire Frame.
2.  **Wind Vector Mathematics Diagram:** A visual representation of a center pixel, the 8 surrounding neighbors, the wind angle, and the dot-product calculation showing how probabilities shift.
3.  **Simulation Side-by-Side (Crucial):**
    *   *Image A:* The pure ML model expanding in a perfect circle (failing to respect wind).
    *   *Image B:* The Hybrid CA model expanding in a narrow, directional cone based on a 80 km/h wind vector.
4.  **Topography Spread Visual:** A 3D or color-coded map showing a fire starting in a valley and rapidly climbing the ridges while struggling to go down the other side.
