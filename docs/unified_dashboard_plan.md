# Unified Interactive Dashboard Implementation Plan

## The Objective
Transform `Simulation2` (the real ML engine) into a Unified Dashboard by pulling in the UI layout from `dashboard` and the advanced mathematical controls from `simulation`.

## Phase 1: File Structure & Scaffolding
1. **Create Unified Directory:** Create a new folder at `Frontend/app/interactive-dashboard`.
2. **Move the Engine:** Copy `Frontend/app/Simulation2/MapSimulation.tsx` and `page.tsx` into this new directory. This ensures we start with the working, FastAPI-connected Leaflet map.
3. **Update Navigation:** Modify `Frontend/components/navigation.tsx` to add a link for "Interactive Dashboard" and comment out the old `dashboard`, `simulation`, and `Simulation2` links to reduce clutter.

## Phase 2: Unifying the Sidebar Layout & Controls
Currently, the `Simulation2` sidebar only has historical events and a basic wind slider. We need to integrate the rich UI from `dashboard` and `simulation`.
1. **Introduce Tabs:** Implement a `<Tabs>` component in the left sidebar with two main panels:
    * **"Historical Analysis"** (For loading real past events and comparing ML prediction vs Ground Truth).
    * **"Sandbox Simulation"** (For the interactive Cellular Automata sandbox).
2. **Port Advanced CA Parameters:** Inside the Sandbox tab, add the granular inputs from the old `simulation` page:
    * *Ignition Threshold* (Slider: 0.0 to 1.0)
    * *Humidity* (Slider: 0% to 100%)
    * *Simulation Steps* (Input field)
    * *(Wind Speed and Direction already exist here but need to be restyled)*.
3. **Map Layers Accordion:** Add an accordion/panel at the bottom of the sidebar inspired by the `dashboard`'s "Map Layers".
    * Instead of the fake toggles in `dashboard`, these toggles will switch the Leaflet base map (e.g., from `Carto Dark` to `OpenStreetMap Satellite`) so users can see the actual vegetation and terrain under the predicted fire.

## Phase 3: Upgrading the CA Physics Engine
The math inside `Simulation2`'s `runCAStep()` is currently a simplified version of the CA math found in the raw `simulation` page. We need to upgrade it.
1. **Integrate Humidity Factor:** Update `runCAStep` so that high humidity structurally lowers the base `prob` of a cell igniting.
2. **Integrate Ignition Threshold:** Allow the user's UI threshold to dynamically act as a cutoff multiplier before a cell flips to state `1` (burning).
3. **Smooth Wind Bias:** Replace the simple dot-product wind check in `Simulation2` with the advanced Cosine-based angular difference logic used in the old `simulateCA` function. This makes the fire fan out more naturally.

## Phase 4: UI Polish & Overlays
1. **Current Conditions Panel:** Port the "Current Conditions" floating card from the `dashboard` (showing Temp/Humidity text). We will pin this to the top-right of the Leaflet map container.
2. **Legend:** Keep the existing comparison legend (Red = Predicted, Green = Actual, Purple = Overlap) but style it to match the sleek dark theme of the new layout.
3. **Wind Animation:** Retain the beautiful SVG wave wind overlay currently in `Simulation2`, ensuring it correctly respects the new wind direction controls.
