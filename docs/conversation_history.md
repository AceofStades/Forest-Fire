# Conversation Log: Moving from Hourly to Daily, and Solving Persistence Bias

## The Problem
We noticed that the model's F1 score was suspiciously perfect (0.998), but the predictions were wildly wrong when simulating the physics of the fire (wind, elevation, etc.). Additionally, the MODIS data was extremely sparse because the satellite only passes over India once or twice per day, making continuous hourly data impossible without introducing major noise. 

The `MODIS_FIRE_T1` (current fire) was passed as an input, and the model was asked to predict the fire at T+1. The easiest way for the UNet to minimize the loss was to learn the identity function: just copy the input fire exactly to the output. Since T and T+1 overlap by 99%, copying the input yielded a 0.998 F1 score instantly. It didn't look at the wind or temperature; it just acted as a mirror.

## The Explored Solutions

### 1. The Hybrid Approach: ML for Fuel + Cellular Automata for Spread (Selected)
Stop asking the ML model to predict the *future fire state*. Instead, train the ML model to look at the static features (vegetation, moisture, urbanization, elevation) and predict a **Static Burn Probability Map (Fuel Model)**. 
Then, use a standard **Cellular Automaton (CA)** explicitly coded with the Rothermel Fire Spread Equations (or similar logical rules). The CA looks at the ML-generated fuel map, checks the exact wind direction/speed, and calculates the math to push the fire front to neighboring pixels. 

### 2. Physics-Informed Neural Networks (PINNs)
Keep using a pure Neural Network, but force it to obey physics by changing the PyTorch loss function. Alongside the `CombinedLoss`, add a "Physics Loss". We use PyTorch to calculate the spatial gradient of the predicted fire and compare it to the wind direction vectors and slope vectors. If the model predicts the fire spreading against the wind, the loss function massively penalizes it.

### 3. Vector-based Front Tracking (Level Set Methods / GNNs)
Instead of treating the map as an image, extract the boundary of the fire as a series of points (a polygon). Use a dense neural network or a Graph Neural Network (GNN) to predict the velocity vector (speed and direction) of each point on the boundary based on the local wind and slope at that exact coordinate.

## Why Our Dataset is like NDWD (Next Day Wildfire Dataset)
Both datasets contain:
1. A massive grid of geographical pixels.
2. Static Variables: Topography (DEM/Slope) and Land Cover.
3. Dynamic Variables: Daily Weather.
4. Target Variable: Satellite fire masks (MODIS/VIIRS) taken as daily/semi-daily snapshots.

Google treats the data as discrete daily snapshots. They predict Day 2's fire mask from Day 1's weather and fire mask. We were trying to predict hourly, using a 24-hour interpolated "Strobe Light" fix. Neural networks are lazy and will just copy the input to the output when predicting hourly steps with 24-hour satellite data.

## The Action Taken
1. **Disabled the `MODIS_FIRE_T1` cheat:** Updated `Model/train.py` to stop feeding the current fire to the UNet (`include_fire_input=False`). The model now acts as a **Burn Susceptibility / Fuel Model**.
2. **Retrained the UNet:** The model achieved a completely legitimate 0.94 F1 score by looking at vegetation, elevation, and weather to predict burn probability (topography and vegetation are highly predictable).
3. **Frontend UI Update:** Changed the UI labels from "Hourly" to "Daily" steps, as each CA step will now effectively represent a day of spread based on the fuel map and wind conditions.
