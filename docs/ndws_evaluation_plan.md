# Next Day Wildfire Spread (NDWS) Dataset: Implementation & Evaluation Plan

## 1. Overview
As part of our comparative research into modeling forest fire dynamics, we are integrating the **Next Day Wildfire Spread (NDWS)** dataset (also commonly known as NDWD, published by Google). 

Unlike our custom dynamic dataset (which provides continuous localized 320x400 gridded data of Uttarakhand, India), the NDWS dataset is constructed from historical US wildfires (e.g., California) and is packaged as thousands of disconnected **64x64 pixel image crops** saved in TensorFlow Record (`.tfrecord`) formats.

This document outlines the detailed architectural plan to ingest this dataset, train our PyTorch models on it, and rigorously compare the results against our existing Custom Hybrid (ML for Fuel + Cellular Automata for Spread) model.

---

## 2. Implementation Plan

### Phase 1: The PyTorch ↔ TensorFlow Data Bridge
PyTorch cannot natively read `.tfrecord` files efficiently. To bridge this ecosystem gap without rewriting our models in TensorFlow, we will build a custom PyTorch `IterableDataset`.

**1. Data Parsing (`dataset.py`):**
*   We will utilize `tf.data.TFRecordDataset` via an isolated TensorFlow import to stream the raw serialized protobuf strings.
*   These strings will be decoded into exactly 12 input features and 1 target feature (`FireMask`).
*   **The 12 Inputs:** 
    *   Topography: `elevation`
    *   Weather: `th` (wind direction), `vs` (wind speed), `tmmn` (min temp), `tmmx` (max temp), `sph` (specific humidity), `pr` (precipitation)
    *   Drought/Fuel Indices: `pdsi` (Palmer Drought Severity Index), `erc` (Energy Release Component), `NDVI` (Vegetation index)
    *   Human Context: `population` (Density)
    *   Fire Context: `PrevFireMask` (Fire footprint on Day 1)
*   These will be stacked into a PyTorch tensor of shape `[Batch, 12, 64, 64]`.

**2. Normalization:**
*   Rather than calculating dynamic percentiles across the dataset, we will strictly adhere to the hardcoded min/max normalization constants published in the original Google NDWS research paper to ensure a 1-to-1 scientific comparison.

### Phase 2: Model Architecture Adaptation
We will reuse our core `UNet` architecture currently located in `Model/training/ndws_dataset/src/models.py`.

**1. Input/Output Constraints:**
*   The UNet will be initialized with `n_channels=12`.
*   The input tensor size will be `64x64`. Because our UNet is fully convolutional and features a Convolutional Block Attention Module (CBAM), it is mathematically resolution-agnostic and will seamlessly accept `64x64` tensors without architectural rewrites.

**2. The Philosophical Difference in Objective:**
*   **Custom Model:** Stripped of `PrevFireMask` to act purely as a **Static Burn Susceptibility (Fuel) Model**.
*   **NDWS Model:** **Must** include `PrevFireMask`. The explicit goal of the NDWS architecture is to let the neural network predict the complex, non-linear advection of the fire front (the spread) by looking at the previous day's fire and determining where it moves on Day 2.

### Phase 3: Training Pipeline
We will create a standalone `train_ndws.py` script.

**1. Loss Function:**
*   NDWS is pathologically imbalanced (fires usually only grow by a few pixels at the borders). We will reuse our **Combined Loss (Focal Loss + Dice Loss)** to force the network's gradients to focus on the expanding boundaries rather than the overwhelming "No Fire" background.

**2. Metrics:**
*   We will calculate Accuracy, Precision, Recall, and F1 Score specifically on the positive class (active fire pixels), evaluating predictions via multiple probability thresholds to find the optimal operating point.

---

## 3. Comparison Methodology

Evaluating an ML model trained on continuous Indian topographical data against one trained on 64x64 US snapshots requires a nuanced, multi-pillar methodology.

### Pillar 1: Quantitative Performance (The Numbers)
*   **Metric:** F1 Score on the validation sets.
*   **Hypothesis:** The NDWS model will likely struggle to break a `0.30 - 0.40` F1 score. Predicting explicit fluid dynamics and spread using only a Day 1 & Day 2 snapshot is mathematically brutal due to the lack of temporal resolution. Conversely, our Custom Model (acting as a fuel map) scored `0.94` because predicting topographical flammability is highly deterministic.

### Pillar 2: Architectural Viability (Black Box vs. Physics CA)
*   **The NDWS Approach (Black Box ML):** The model ingests Day 1 and strictly outputs Day 2. It is monolithic. If a user asks, *"What if the wind changes at 2:00 PM?"*, the NDWS model cannot answer because it only processes 24-hour pre-calculated blocks.
*   **Our Custom Approach (Hybrid):** Our ML model maps the fuel layout, and a frontend React Cellular Automaton handles the temporal physics. If the user shifts the wind at 2:00 PM in the UI, the CA instantly recalculates the spread. 
*   **Conclusion:** We will evaluate the rigidity of a pure ML spread predictor (NDWS) versus the flexibility of an ML+Math simulation engine.

### Pillar 3: Generalization & Deployment Reality
*   **Data Availability:** The NDWS dataset relies heavily on `pdsi` (Palmer Drought Severity Index) and `erc` (Energy Release Component). These are highly specific, complex data products calculated locally by US Forestry agencies. 
*   **Conclusion:** We will document that our Custom Model relies solely on globally accessible ERA5 satellite data and simple DEM maps. This makes our custom approach globally scalable and immediately deployable in developing regions (like the Himalayas), whereas an NDWS-style architecture is heavily bottlenecked by US-centric advanced data pipelines.