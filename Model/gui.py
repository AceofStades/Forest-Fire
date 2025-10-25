import streamlit as st
import xarray as xr
import pandas as pd
import os
# The following line was the error and has been removed:
# from xarray.core.coordinates import Coordinate

st.set_page_config(layout="wide")

# --- 1. CONFIGURATION ---
FINAL_STACK_PATH = "dataset/final_feature_stack.nc"
FINAL_STACK_PATH1 = "dataset/final_feature_stack1.nc"
FINAL_STACK_PATH2 = "dataset/final_feature_stack2.nc"
MODIS_CSV_PATH = "dataset/MODIS/modis_2016_India.csv"


# --- 2. Function to Safely Load and Sample Xarray Data ---
def safe_xarray_head(
    path, time_slice=slice(0, 5), lat_slice=slice(0, 5), lon_slice=slice(0, 5)
):
    """Loads a tiny, explicit slice of data to prevent memory overload."""
    try:
        # Load the file lazily
        ds = xr.open_dataset(path)

        # CRITICAL FIX: Sample the dataset lazily before converting to DataFrame.
        # This loads only a minimal subset of data.
        ds_sampled = ds.isel(
            valid_time=time_slice, latitude=lat_slice, longitude=lon_slice
        )

        # Now, convert only this tiny sample to a DataFrame.
        df = ds_sampled.to_dataframe().reset_index()

        # Clean up the valid_time format for display clarity
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


# --- 3. Streamlit Display ---
st.title("Forest Fire Modeling Data Diagnostics")
st.caption("Showing diagnostic sample slices only, to avoid memory overload.")
st.divider()


## Final Feature Stack Display
st.subheader("1. Final Feature Stack (1km Fused Data)")
st.caption(
    f"Showing the first 5 time steps (rows) and first 5x5 pixels (columns) from: {FINAL_STACK_PATH}"
)

# Load only the safe sample slice
df_final_stack_head = safe_xarray_head(FINAL_STACK_PATH)
st.dataframe(df_final_stack_head)


## Final Feature Stack Display
st.subheader("1. Final Feature Stack (1km Fused Data)")
st.caption(
    f"Showing the first 5 time steps (rows) and first 5x5 pixels (columns) from: {FINAL_STACK_PATH1}"
)

# Load only the safe sample slice
df_final_stack_head = safe_xarray_head(FINAL_STACK_PATH1)
st.dataframe(df_final_stack_head)

## Final Feature Stack Display
st.subheader("1. Final Feature Stack (1km Fused Data)")
st.caption(
    f"Showing the first 5 time steps (rows) and first 5x5 pixels (columns) from: {FINAL_STACK_PATH1}"
)

# Load only the safe sample slice
df_final_stack_head = safe_xarray_head(FINAL_STACK_PATH2)
st.dataframe(df_final_stack_head)

## Optional Visualizations (Keeping them small)
st.divider()
st.subheader("2. Raster Visualizations & Input Data Heads")

# --- Load Original MODIS (as it's a small file and helpful for checks) ---
try:
    df_orig_modis = pd.read_csv(MODIS_CSV_PATH)
    st.subheader("Original MODIS Point Data (Head)")
    st.dataframe(df_orig_modis.head())
except:
    st.warning(f"Could not load original MODIS CSV at {MODIS_CSV_PATH}")

st.image("dataset/DEM/dem_plot.png", caption="DEM (Elevation) Sample")
st.image(
    "dataset/GHS/ghs_downsampled_plot.png", caption="GHS (Human Settlement) Sample"
)
st.image("dataset/LULC/lulc.png", caption="LULC (Land Cover) Sample")
