# Architecture & Design Decisions: The "Persistence Bias" Resolution

This document tracks the crucial pivot made to resolve the "Identity Trap" (Persistence Bias) that caused the UNet and ConvLSTM models to falsely achieve 0.99 F1 scores without learning physical fire dynamics.

## 1. The Core Issue: The "Strobe Light" Effect & Identity Leak
The raw dataset relies on MODIS satellite data, which only provides a snapshot of the fire 1-2 times per day. In our hourly dataset array, this meant the fire was entirely static for 23 hours a day, jumping abruptly once the satellite passed over.

*   **Attempt 1 (Synthetic Hourly Spread):** We initially created `interpolate_fire_morphology.py` to synthetically grow the fire hour-by-hour to bridge the 24-hour satellite gaps. 
*   **The Model Failure (Predicting Delta):** We asked the network to predict *only* the newly ignited pixels (the Delta) between T and T+1. Because the interpolated expansion for a single hour was microscopically thin, it was impossible for the ML model to learn it against the massive class imbalance. F1 scores collapsed to 0.000.
*   **The Model Failure (Predicting T+1 with T as input):** We then changed the objective to predict the *entire* fire state at T+1, passing the current fire map (`MODIS_FIRE_T1`) as a 14th input channel. Because T and T+1 overlap by >99.9%, the UNet learned an "Identity Function." It simply copied the 14th input channel directly to the output. It ignored wind, slope, and temperature entirely, instantly achieving a 0.99 F1 score while fundamentally failing to learn any physics.

## 2. The Solution: ML for Fuel + CA for Spread
Because the temporal resolution of satellites is fundamentally too low to teach a pure neural network fluid dynamics hour-by-hour, we restructured the architecture to mimic Google's Next Day Wildfire Dataset (NDWD) philosophy while supporting an interactive 60fps frontend simulation.

### Step 1: Strip the Temporal Cheat from the ML Model
We updated `Model/train.py` to set `include_fire_input=False`. 
*   **The Pivot:** The UNet is **no longer a temporal predictor**. It is now a **Static Burn Susceptibility (Fuel) Model**. 
*   **Input:** 13 channels (Weather, Vegetation, Topography, Burn Scar). It does *not* know where the fire currently is.
*   **Output:** A single 2D probability matrix representing the inherent flammability/susceptibility of every pixel based on the static landscape and current weather.

### Step 2: Empower the Physics Engine (Cellular Automata)
Since the ML model handles the complex, non-linear pattern recognition of the terrain, we moved the actual temporal advection (movement over time) out of PyTorch and into a hard-coded mathematical engine.
*   **The Integration:** The React frontend (and Python backend) runs a standard Cellular Automaton.
*   **How it works:** The user drops a fire ignition point on the map. The CA algorithm looks at the underlying ML-generated Susceptibility Map to see if the pixel can burn, and then uses explicit mathematical rules (based on the user's Wind Speed and Direction sliders) to push the fire to neighboring pixels. 

## 3. Results and Current State
This hybrid approach successfully bypasses the limitations of sparse satellite data:
1.  **Valid Metrics:** By removing the `MODIS_FIRE_T1` cheat channel, the UNet is forced to actually synthesize the weather and elevation data. It currently achieves a highly legitimate `0.94` F1 score on predicting burn probability based on environmental factors alone.
2.  **Physics Compliance:** Because the temporal spread is driven by the CA engine in the frontend, the simulation perfectly obeys wind direction and speed constraints without suffering from the ML model "teleporting" fires.