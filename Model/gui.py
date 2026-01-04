import os

import pandas as pd
import streamlit as st
import xarray as xr

st.set_page_config(layout="wide")

FINAL_STACK_PATH = "dataset/final_feature_stack.nc"
FINAL_STACK_PATH1 = "dataset/final_feature_stack1.nc"
FINAL_STACK_PATH2 = "dataset/final_feature_stack2.nc"
FINAL_MASTER_PATH = "dataset/final_feature_stack_MASTER.nc"

MODIS_OG = "dataset/MODIS/modis_2016_India.csv"
MODIS_FINAL = "dataset/MODIS/final-modis.csv"
ERA5_PATH = "dataset/ERA5-Land/era5-april/data_0.nc"
ERA5_PATH_RE = "dataset/ERA5-Land/final-era5_rechunked.nc"


def safe_xarray_head(
    path, time_slice=slice(0, 5), lat_slice=slice(0, 5), lon_slice=slice(0, 5)
):
    """Loads a tiny, explicit slice of data to prevent memory overload."""
    try:
        ds = xr.open_dataset(path)

        ds_sampled = ds.isel(
            valid_time=time_slice, latitude=lat_slice, longitude=lon_slice
        )

        df = ds_sampled.to_dataframe().reset_index()

        if "valid_time" in df.columns:
            df["valid_time"] = df["valid_time"].dt.strftime("%Y-%m-%d %H:%M")

        return df

    except Exception as e:
        return pd.DataFrame(
            {
                "Error": [
                    f"Could not safely load xarray data: {e}. Check file path/integrity."
                ]
            }
        )


st.title("Forest Fire Project Diagnostics")
st.caption("Showing diagnostic sample slices only, to avoid memory overload.")
st.divider()

st.subheader("0. Final MASTER Feature Stack (Intersection Crop)")
st.caption(
    f"This is the production-ready dataset with zero edge padding: {FINAL_MASTER_PATH}"
)
df_master = safe_xarray_head(FINAL_MASTER_PATH)
st.dataframe(df_master)

st.divider()

st.subheader("1. Previous Iterations (Diagnostic History)")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"Stack V0: {FINAL_STACK_PATH}")
    st.dataframe(safe_xarray_head(FINAL_STACK_PATH))

with col2:
    st.caption(f"Stack V1: {FINAL_STACK_PATH1}")
    st.dataframe(safe_xarray_head(FINAL_STACK_PATH1))

with col3:
    st.caption(f"Stack V2: {FINAL_STACK_PATH2}")
    st.dataframe(safe_xarray_head(FINAL_STACK_PATH2))

st.divider()
st.subheader("2. Raster Visualizations & Input Data Heads")

try:
    df_orig_modis = pd.read_csv(MODIS_OG)
    st.subheader("Original MODIS Point Data (Head)")
    st.dataframe(df_orig_modis.head(100))
except:
    st.warning(f"Could not load original MODIS CSV at {MODIS_OG}")

try:
    df_final_modis = pd.read_csv(MODIS_FINAL)
    st.subheader("Final MODIS Point Data (Head)")
    st.dataframe(df_final_modis.head(100))
except:
    st.warning(f"Could not load original MODIS CSV at {MODIS_FINAL}")

st.image("dataset/DEM/dem_plot.png", caption="DEM (Elevation) Sample")
st.image(
    "dataset/GHS/ghs_downsampled_plot.png", caption="GHS (Human Settlement) Sample"
)
st.image("dataset/LULC/lulc.png", caption="LULC (Land Cover) Sample")


st.subheader("Original ERA5-Land")
st.caption(
    f"Showing the first 5 time steps (rows) and first 5x5 pixels (columns) from: {ERA5_PATH}"
)
df_era5 = safe_xarray_head(ERA5_PATH)
st.dataframe(df_era5)

st.subheader("Final ERA5-Land (Rechunked)")
st.caption(
    f"Showing the first 5 time steps (rows) and first 5x5 pixels (columns) from: {ERA5_PATH_RE}"
)
df_era5_final = safe_xarray_head(ERA5_PATH_RE)
st.dataframe(df_era5_final)
