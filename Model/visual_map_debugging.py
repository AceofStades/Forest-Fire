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
    # Use h5netcdf for stability on large files
    master = xr.open_dataset(MASTER_PATH, engine="h5netcdf", chunks={})
    era5 = xr.open_dataset(ERA5_SOURCE_PATH, engine="h5netcdf", chunks={})
    return master, era5


def sample_raster_neighbor(path, lat, lon):
    try:
        with rasterio.open(path) as src:
            row, col = rowcol(src.transform, lon, lat)
            window = src.read(1)[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2]
            return np.nanmin(window), np.nanmax(window), src.read(1)[row, col]
    except:
        return np.nan, np.nan, np.nan


def sample_era5_neighbor(ds, lat, lon, var_name):
    try:
        lat_idx = np.abs(ds.latitude.values - lat).argmin()
        lon_idx = np.abs(ds.longitude.values - lon).argmin()
        window = (
            ds[var_name]
            .isel(
                valid_time=0,
                latitude=slice(max(0, lat_idx - 1), lat_idx + 2),
                longitude=slice(max(0, lon_idx - 1), lon_idx + 2),
            )
            .compute()
        )
        v_min, v_max = window.min().values, window.max().values
        v_near = (
            ds[var_name]
            .sel(latitude=lat, longitude=lon, method="nearest")
            .isel(valid_time=0)
            .values
        )
        return float(v_min), float(v_max), float(v_near)
    except:
        return np.nan, np.nan, np.nan


st.title("Final Master Dataset Tally Tool")

master_ds, era5_ds = load_datasets()
lats, lons = master_ds.latitude.values, master_ds.longitude.values

col_map, col_data = st.columns([1, 1.2])

with col_map:
    st.subheader("Master Grid Selection")
    # Default to center of dataset
    m = folium.Map(location=[lats.mean(), lons.mean()], zoom_start=8)
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

        # Pull data from the Master 1km Stack
        pixel_master = (
            master_ds.sel(latitude=lat, longitude=lon, method="nearest")
            .isel(valid_time=0)
            .compute()
        )

        tally_data = []

        # 1. ERA5 Comparison with 9km Neighborhood Debug
        for v in era5_ds.data_vars:
            s_min, s_max, s_near = sample_era5_neighbor(era5_ds, lat, lon, v)
            m_val = float(pixel_master[v].values)
            within_range = m_val >= s_min - 1e-4 and m_val <= s_max + 1e-4
            status = "✅ Local Match" if within_range else "❌ Out of Bounds"

            tally_data.append(
                {
                    "Variable": v,
                    "Master (1km)": f"{m_val:.4f}",
                    "Source Range (9km 3x3)": f"[{s_min:.4f} to {s_max:.4f}]",
                    "Status": status,
                }
            )

        # 2. Static Comparison (DEM, LULC, GHS)
        for name, path in STATIC_SOURCES.items():
            s_min, s_max, s_near = sample_raster_neighbor(path, lat, lon)
            m_val = float(pixel_master[name].values)

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
                    "Master (1km)": f"{m_val:.2f}",
                    "Source Range (3x3)": f"[{s_min:.1f} to {s_max:.1f}]",
                    "Status": status,
                }
            )

        st.table(pd.DataFrame(tally_data))

        # 3. MODIS Proximity Search
        st.divider()
        st.subheader("Nearest MODIS Fire Events (CSV Ground Truth)")
        if os.path.exists(MODIS_CSV):
            df_modis = pd.read_csv(MODIS_CSV)
            # Simple Euclidean distance for proximity check
            df_modis["dist"] = np.sqrt(
                (df_modis["latitude"] - lat) ** 2 + (df_modis["longitude"] - lon) ** 2
            )
            nearby_fires = df_modis.nsmallest(5, "dist")[
                ["acq_date", "latitude", "longitude", "confidence", "dist"]
            ]
            st.dataframe(nearby_fires)

            # Highlight if the current Master cell shows fire
            m_fire = float(pixel_master["MODIS_FIRE_T1"].values)
            if m_fire > 0:
                st.success(
                    f"Confirmed: Master Stack shows FIRE (1.0) at this coordinate."
                )
            else:
                st.info("Master Stack shows NO FIRE (0.0) at this coordinate.")
        else:
            st.error("MODIS CSV not found for proximity check.")

        st.warning(
            "Diagnostic: If the Master value is within the Source Range, the interpolation is accurate."
        )
else:
    with col_data:
        st.info("Click a point on the map to begin the spatial audit.")
