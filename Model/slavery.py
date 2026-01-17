import os

import folium
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import xarray as xr
from rasterio.transform import rowcol
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

MASTER_PATH = "dataset/final_feature_stack_MASTER.nc"
ERA5_SOURCE_PATH = "dataset/ERA5-Land/final-era5_rechunked.nc"
STATIC_SOURCES = {
    "DEM": "dataset/resampled-fix/dem_resampled.tif",
    "LULC": "dataset/resampled-fix/lulc_resampled.tif",
    "GHS_BUILT": "dataset/resampled-fix/ghs_resampled.tif",
}
MODIS_CSV = "dataset/MODIS/final-modis.csv"


@st.cache_resource
def load_datasets():
    master = xr.open_dataset(MASTER_PATH, engine="h5netcdf", chunks={})
    era5 = xr.open_dataset(ERA5_SOURCE_PATH, engine="h5netcdf", chunks={})
    return master, era5


def sample_raster_neighbor(path, lat, lon):
    try:
        with rasterio.open(path) as src:
            row, col = rowcol(src.transform, lon, lat)
            # We take a 3x3 window around the original source
            window = src.read(1)[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2]
            return np.nanmin(window), np.nanmax(window), src.read(1)[row, col]
    except:
        return np.nan, np.nan, np.nan


st.title("Final Master Dataset Tally Tool (Spatial Debugger)")

master_ds, era5_ds = load_datasets()
lats, lons = master_ds.latitude.values, master_ds.longitude.values

col_map, col_data = st.columns([1, 1.2])

with col_map:
    st.subheader("Master Grid Selection")
    m = folium.Map(
        location=[lat, lon] if "lat" in locals() else [lats.mean(), lons.mean()],
        zoom_start=10,
    )
    folium.Rectangle(
        bounds=[[lats.min(), lons.min()], [lats.max(), lons.max()]],
        color="red",
        weight=2,
        fill=False,
    ).add_to(m)
    click_data = st_folium(m, width=600, height=600)

if click_data.get("last_clicked"):
    lat = click_data["last_clicked"]["lat"]
    lon = click_data["last_clicked"]["lng"]

    with col_data:
        st.subheader(f"Results for: {lat:.4f}, {lon:.4f}")

        pixel_master = (
            master_ds.sel(latitude=lat, longitude=lon, method="nearest")
            .isel(valid_time=0)
            .compute()
        )
        pixel_era5 = (
            era5_ds.sel(latitude=lat, longitude=lon, method="nearest")
            .isel(valid_time=0)
            .compute()
        )

        tally_data = []

        # ERA5 Check
        for v in era5_ds.data_vars:
            tally_data.append(
                {
                    "Variable": v,
                    "Master Value": f"{pixel_master[v].values:.4f}",
                    "Source (Nearest)": f"{pixel_era5[v].values:.4f}",
                    "Note": "Interpolated",
                }
            )

        # Static Check with Neighborhood Debug
        for name, path in STATIC_SOURCES.items():
            s_min, s_max, s_near = sample_raster_neighbor(path, lat, lon)
            m_val = pixel_master[name].values

            # If Master is within the range of original neighbors, it's correct!
            status = (
                "✅ Local Match"
                if (m_val >= s_min and m_val <= s_max)
                else "❌ Spatial Shift"
            )
            if name == "LULC" and m_val == s_near:
                status = "✅ Perfect"
            if s_near == 252:
                status = "⚠️ Fixed NoData"

            tally_data.append(
                {
                    "Variable": name,
                    "Master Value": f"{m_val:.2f}",
                    "Source (Nearest)": f"{s_near:.2f}",
                    "Source Range (3x3)": f"[{s_min:.1f} to {s_max:.1f}]",
                    "Status": status,
                }
            )

        st.table(pd.DataFrame(tally_data))
        st.warning(
            "**Diagnostic:** If 'Master Value' is inside the 'Source Range (3x3)', the data is accurate. It just means the 1km grid center doesn't align perfectly with the original sensor pixel."
        )
