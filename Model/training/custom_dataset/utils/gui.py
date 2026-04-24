import os

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

st.set_page_config(layout="wide")

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


# Helper to join paths relative to the script location
def get_path(rel_path):
    return os.path.join(BASE_DIR, rel_path)


FINAL_MASTER_PATH = get_path("dataset/final_feature_stack_MASTER.nc")
MODIS_OG = get_path("dataset/MODIS/modis_2016_India.csv")
MODIS_FINAL = get_path("dataset/MODIS/final-modis.csv")
ERA5_PATH = get_path("dataset/ERA5-Land/era5-april/data_0.nc")
ERA5_PATH_RE = get_path("dataset/ERA5-Land/final-era5_rechunked.nc")

# --- HELPER FUNCTIONS ---


def parse_and_save_csv(txt_path, csv_path):
    """Reads a metrics text file and saves/returns a CSV DataFrame."""
    if not os.path.exists(txt_path):
        return None

    data = {}
    try:
        with open(txt_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if key.lower() == "model":
                        continue  # Skip model name row
                    try:
                        data[key] = float(val)
                    except ValueError:
                        data[key] = val

        df = pd.DataFrame.from_dict(data, orient="index", columns=["Score"])
        df.to_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error parsing {txt_path}: {e}")
        return None


def get_data_safe(
    path, t_slice=slice(0, 1), lat_slice=slice(0, 5), lon_slice=slice(0, 5)
):
    if not os.path.exists(path):
        return pd.DataFrame({"Error": [f"File not found: {path}"]})
    try:
        with xr.open_dataset(path, engine="h5netcdf", chunks={}) as ds:
            subset = ds.isel(
                valid_time=t_slice, latitude=lat_slice, longitude=lon_slice
            ).compute()
            df = subset.to_dataframe().reset_index()
            if "valid_time" in df.columns:
                df["valid_time"] = df["valid_time"].dt.strftime("%Y-%m-%d %H:%M")
            return df
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


def get_dense_samples(path, var_name, num_samples=3):
    if not os.path.exists(path):
        return []

    samples = []
    try:
        with xr.open_dataset(path, engine="h5netcdf", chunks={}) as ds:
            if var_name not in ds.data_vars:
                return [pd.DataFrame({"Error": [f"{var_name} not found in dataset"]})]

            data_var = ds[var_name]

            # Fix for static variables (GHS, LULC) lacking valid_time
            if "valid_time" in data_var.dims:
                if var_name == "MODIS_FIRE_T1":
                    search_data = data_var.compute()
                else:
                    search_data = data_var.isel(valid_time=0).compute()
            else:
                search_data = data_var.compute()

            raw_values = search_data.values
            indices = np.argwhere(raw_values > 0)

            if len(indices) == 0:
                return [
                    pd.DataFrame({"Info": [f"No non-zero values found for {var_name}"]})
                ]

            valid_clusters = []
            for idx in indices:
                # Handle 2D (lat, lon) vs 3D (time, lat, lon) indices
                if len(idx) == 3:
                    t, lat, lon = idx
                    window = raw_values[
                        t, max(0, lat - 1) : lat + 2, max(0, lon - 1) : lon + 2
                    ]
                else:
                    lat, lon = idx
                    window = raw_values[
                        max(0, lat - 1) : lat + 2, max(0, lon - 1) : lon + 2
                    ]

                if np.count_nonzero(window) > 1:
                    valid_clusters.append(idx)

                if len(valid_clusters) >= 200:
                    break

            if not valid_clusters:
                valid_clusters = indices[:num_samples]
            else:
                step = max(1, len(valid_clusters) // num_samples)
                valid_clusters = [
                    valid_clusters[i] for i in range(0, len(valid_clusters), step)
                ][:num_samples]

            for idx in valid_clusters:
                if len(idx) == 3:
                    t_idx, lat_idx, lon_idx = idx
                else:
                    t_idx, lat_idx, lon_idx = 0, idx[0], idx[1]

                subset = ds.isel(
                    valid_time=slice(t_idx, t_idx + 1),
                    latitude=slice(max(0, lat_idx - 3), lat_idx + 4),
                    longitude=slice(max(0, lon_idx - 3), lon_idx + 4),
                ).compute()
                samples.append(subset.to_dataframe().reset_index())

        return samples
    except Exception as e:
        return [pd.DataFrame({"Error": [str(e)]})]


# --- GUI LAYOUT ---

st.title("Forest Fire Project Diagnostics")
st.divider()

st.subheader("In-depth Model Hyperparameters")

# Hyperparameter Data
hyperparams_data = {
    "Model": [
        "Legacy UNet",
        "ConvLSTM",
        "Hybrid Model",
        "Custom Dataset (UNet/ConvLSTM)",
    ],
    "Epochs": [25, 15, 50, 30],
    "Batch Size": [8, 4, 32, 16],
    "Accumulation Steps": [1, 1, 1, 8],
    "Learning Rate": ["1e-3", "1e-3", "1e-4", "1e-3"],
    "Optimizer": ["Adam", "Adam", "Adam", "AdamW"],
    "Weight Decay": ["0", "0", "1e-4", "1e-4"],
    "Scheduler": ["None", "None", "None", "CosineAnnealingWarmRestarts"],
    "Loss Function": [
        "BCEWithLogitsLoss / FocalLoss (implied)",
        "BCEWithLogitsLoss",
        "BCEWithLogitsLoss + Dice",
        "CombinedLoss (Focal + Dice / Tversky)",
    ],
    "BCE pos_weight": ["-", "-", "20.0", "500-3000 (tuned)"],
    "Sequence Length": ["-", "3", "3", "4"],
    "Hidden Dims": ["-", "[32, 32, 32]", "[64] (LSTM part)", "[64, 64]"],
}
df_hyperparams = pd.DataFrame(hyperparams_data)
st.dataframe(df_hyperparams, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Model Performance Comparison")

# Parse and create CSVs
m1_txt = get_path("assets/best_fire_unet_results.txt")
m1_csv = get_path("assets/best_fire_unet_results.csv")
df_m1 = parse_and_save_csv(m1_txt, m1_csv)

m2_txt = get_path("assets/best_convlstm_results.txt")
m2_csv = get_path("assets/best_convlstm_results.csv")
df_m2 = parse_and_save_csv(m2_txt, m2_csv)

m3_txt = get_path("assets/best_hybrid_model_results.txt")
m3_csv = get_path("assets/best_hybrid_model_results.csv")
df_m3 = parse_and_save_csv(m3_txt, m3_csv)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Legacy UNet")
    if df_m1 is not None:
        st.dataframe(df_m1.style.format("{:.4f}"))
    else:
        st.info("No data")

with col2:
    st.markdown("#### ConvLSTM")
    if df_m2 is not None:
        st.dataframe(df_m2.style.format("{:.4f}"))
    else:
        st.info("No data")

with col3:
    st.markdown("#### Hybrid Model")
    if df_m3 is not None:
        st.dataframe(df_m3.style.format("{:.4f}"))
    else:
        st.info("No data")

st.divider()

st.subheader("Model Evaluation Plots")
tab1, tab2, tab3 = st.tabs(["Legacy UNet", "ConvLSTM", "Hybrid"])


def display_model_tab(model_name, display_name):
    st.markdown(f"### {display_name}")

    col1, col2 = st.columns(2)

    # Paths
    curves_path = get_path(f"assets/{model_name}_curves.png")
    visual_path = get_path(f"assets/{model_name}_visual.png")

    # Display Curves
    if os.path.exists(curves_path):
        col1.image(curves_path, caption="Performance Curves")
    else:
        col1.info(f"No curves plot found at {curves_path}.")

    # Display Visual Sample
    if os.path.exists(visual_path):
        col2.image(visual_path, caption="Sample Prediction")
    else:
        col2.info(f"No visual sample found at {visual_path}.")


with tab1:
    display_model_tab("best_fire_unet", "Legacy UNet (Spatial Only)")

with tab2:
    display_model_tab("best_convlstm", "ConvLSTM (Spatiotemporal Sequence)")

with tab3:
    display_model_tab("best_hybrid_model", "Hybrid Model (ConvLSTM + UNet)")

st.divider()

if os.path.exists(FINAL_MASTER_PATH):
    st.subheader("Master Dataset: Top-Left Corner")
    st.dataframe(get_data_safe(FINAL_MASTER_PATH))

    st.divider()

    st.subheader("Dense Data Clusters")

    for var in ["MODIS_FIRE_T1", "GHS_BUILT", "LULC"]:
        st.write(f"Variable: {var}")
        samples = get_dense_samples(FINAL_MASTER_PATH, var, num_samples=3)
        cols = st.columns(len(samples))
        for i, sample_df in enumerate(samples):
            with cols[i]:
                st.dataframe(sample_df)
        st.write("---")

st.divider()

try:
    df_orig_modis = pd.read_csv(MODIS_OG)
    st.subheader("Original MODIS Point Data")
    st.dataframe(df_orig_modis.head(50))
except:
    pass

dem_path = get_path("dataset/DEM/dem_plot.png")
if os.path.exists(dem_path):
    st.image(dem_path, caption="Elevation")

ghs_path = get_path("dataset/GHS/ghs_downsampled_plot.png")
if os.path.exists(ghs_path):
    st.image(ghs_path, caption="Settlements")

lulc_path = get_path("dataset/LULC/lulc.png")
if os.path.exists(lulc_path):
    st.image(lulc_path, caption="Land Use")

st.subheader("Original ERA5-Land")
st.dataframe(get_data_safe(ERA5_PATH))

st.subheader("Final ERA5-Land (Rechunked)")
st.dataframe(get_data_safe(ERA5_PATH_RE))
