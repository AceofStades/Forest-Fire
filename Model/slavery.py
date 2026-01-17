import os

import folium
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import streamlit as st
import xarray as xr
from rasterio.transform import rowcol
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Data Validation Tally Tool")

MASTER_PATH = "dataset/final_feature_stack_MASTER.nc"
SOURCE_FILES = {
    "DEM": "dataset/resampled-fix/dem_resampled.tif",
    "LULC": "dataset/resampled-fix/lulc_resampled.tif",
    "GHS": "dataset/resampled-fix/ghs_resampled.tif",
    "MODIS": "dataset/MODIS/final-modis.csv",
}


@st.cache_resource
def load_master():
    return xr.open_dataset(MASTER_PATH, engine="h5netcdf", chunks={})


def get_pixel_data(ds, lat, lon):
    try:
        pixel = ds.sel(latitude=lat, longitude=lon, method="nearest").compute()
        return pixel.to_dataframe().reset_index()
    except:
        return None


def sample_source_raster(path, lat, lon):
    try:
        with rasterio.open(path) as src:
            row, col = rowcol(src.transform, lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                return src.read(1)[row, col]
        return "Out of Bounds"
    except:
        return "Error"


st.title("Geospatial Tally & Validation Tool")

if not os.path.exists(MASTER_PATH):
    st.error("Master dataset missing.")
    st.stop()

ds_master = load_master()
lats = ds_master.latitude.values
lons = ds_master.longitude.values
center_lat, center_lon = lats.mean(), lons.mean()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Master Merged Data (1km Grid)")
    m1 = folium.Map(
        location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap"
    )

    # Grid Overlay logic
    grid_bounds = [[lats.min(), lons.min()], [lats.max(), lons.max()]]
    folium.Rectangle(bounds=grid_bounds, color="red", weight=2, fill=False).add_to(m1)

    output1 = st_folium(m1, width=700, height=500, key="master_map")

with col_right:
    st.subheader("Original Source Data")
    tab_names = list(SOURCE_FILES.keys())
    tabs = st.tabs(tab_names)

    for i, tab in enumerate(tabs):
        with tab:
            var_name = tab_names[i]
            path = SOURCE_FILES[var_name]

            m2 = folium.Map(
                location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap"
            )

            if var_name == "MODIS":
                if os.path.exists(path):
                    df_modis = pd.read_csv(path).head(500)
                    for _, row in df_modis.iterrows():
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=3,
                            color="orange",
                            fill=True,
                        ).add_to(m2)
            else:
                folium.Rectangle(
                    bounds=grid_bounds, color="blue", weight=1, fill=False
                ).add_to(m2)

            st_folium(m2, width=700, height=500, key=f"source_map_{var_name}")

if output1.get("last_clicked"):
    clicked_lat = output1["last_clicked"]["lat"]
    clicked_lon = output1["last_clicked"]["lng"]

    st.divider()
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.write(f"Querying Master Pixel at: {clicked_lat:.4f}, {clicked_lon:.4f}")
        master_data = get_pixel_data(ds_master, clicked_lat, clicked_lon)
        if master_data is not None:
            st.dataframe(master_data)

    with res_col2:
        st.write("Source Comparison (Raw File Extraction)")
        comparison = []
        for name, path in SOURCE_FILES.items():
            if name != "MODIS":
                val = sample_source_raster(path, clicked_lat, clicked_lon)
                comparison.append({"Feature": name, "Original Value": val})
        st.table(pd.DataFrame(comparison))
else:
    st.info("Click on the Master Map to tally specific grid coordinates.")
