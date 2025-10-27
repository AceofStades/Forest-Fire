import streamlit as st
import xarray as xr
import pandas as pd
import os

st.set_page_config(layout="wide")

FINAL_STACK_PATH = "dataset/final_feature_stack.nc"
FINAL_STACK_PATH1 = "dataset/final_feature_stack1.nc"
FINAL_STACK_PATH2 = "dataset/final_feature_stack2.nc"
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
                    f"Could not safely load final stack: {e}. Check file path/integrity."
                ]
            }
        )


st.caption("Showing diagnostic sample slices only, to avoid memory overload.")
st.divider()


st.subheader("1. Final Feature Stack (1km Fused Data)")
st.caption(
    f"Showing the first 5 time steps (rows) and first 5x5 pixels (columns) from: {FINAL_STACK_PATH}"
)

df_final_stack_head = safe_xarray_head(FINAL_STACK_PATH)
st.dataframe(df_final_stack_head)


st.caption(
    f"Showing the first 5 time steps (rows) and first 5x5 pixels (columns) from: {FINAL_STACK_PATH1}"
)

df_final_stack_head = safe_xarray_head(FINAL_STACK_PATH1)
st.dataframe(df_final_stack_head)

st.caption(
    f"Showing the first 5 time steps (rows) and first 5x5 pixels (columns) from: {FINAL_STACK_PATH1}"
)

df_final_stack_head = safe_xarray_head(FINAL_STACK_PATH2)
st.dataframe(df_final_stack_head)

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

st.subheader("Final ERA5-Land")
st.caption(
    f"Showing the first 5 time steps (rows) and first 5x5 pixels (columns) from: {ERA5_PATH}"
)

df_era5_final = safe_xarray_head(ERA5_PATH_RE)
st.dataframe(df_era5_final)
