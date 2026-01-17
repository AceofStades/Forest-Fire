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


def sample_raster(path, lat, lon):
    try:
        with rasterio.open(path) as src:
            row, col = rowcol(src.transform, lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                return src.read(1)[row, col]
        return np.nan
    except:
        return np.nan


st.title("Final Master Dataset Tally Tool")

master_ds, era5_ds = load_datasets()
lats, lons = master_ds.latitude.values, master_ds.longitude.values

col_map, col_data = st.columns([1.5, 1])

with col_map:
    st.subheader("Master Grid Selection")
    m = folium.Map(location=[lats.mean(), lons.mean()], zoom_start=8)
    folium.Rectangle(
        bounds=[[lats.min(), lons.min()], [lats.max(), lons.max()]],
        color="red",
        weight=2,
        fill=False,
    ).add_to(m)

    click_data = st_folium(m, width=800, height=600)

if click_data.get("last_clicked"):
    lat = click_data["last_clicked"]["lat"]
    lon = click_data["last_clicked"]["lng"]

    with col_data:
        st.subheader(f"Coordinates: {lat:.4f}, {lon:.4f}")

        # 1. Fetch Master Values
        pixel_master = (
            master_ds.sel(latitude=lat, longitude=lon, method="nearest")
            .isel(valid_time=0)
            .compute()
        )

        # 2. Fetch Original ERA5 Values
        pixel_era5 = (
            era5_ds.sel(latitude=lat, longitude=lon, method="nearest")
            .isel(valid_time=0)
            .compute()
        )

        # 3. Compile Tally Table
        tally_data = []

        # Add ERA5 Variables
        era5_vars = [v for v in era5_ds.data_vars]
        for v in era5_vars:
            tally_data.append(
                {
                    "Variable": v,
                    "Master (Final)": f"{pixel_master[v].values:.6f}",
                    "Source (Original)": f"{pixel_era5[v].values:.6f}",
                    "Match": "✅"
                    if np.isclose(
                        pixel_master[v].values, pixel_era5[v].values, atol=1e-4
                    )
                    else "⚠️ Interp",
                }
            )

        # Add Static Variables
        for name, path in STATIC_SOURCES.items():
            s_val = sample_raster(path, lat, lon)
            m_val = pixel_master[name].values
            tally_data.append(
                {
                    "Variable": name,
                    "Master (Final)": f"{m_val:.2f}",
                    "Source (Original)": f"{s_val:.2f}",
                    "Match": "✅"
                    if np.isclose(m_val, s_val, atol=1e-2)
                    else "⚠️ Resampled",
                }
            )

        st.table(pd.DataFrame(tally_data))

        # 4. MODIS Proximity Check
        st.subheader("Nearest MODIS Fire Events")
        if os.path.exists(MODIS_CSV):
            df_modis = pd.read_csv(MODIS_CSV)
            df_modis["dist"] = np.sqrt(
                (df_modis["latitude"] - lat) ** 2 + (df_modis["longitude"] - lon) ** 2
            )
            nearby_fires = df_modis.nsmallest(5, "dist")[
                ["acq_date", "latitude", "longitude", "confidence"]
            ]
            st.dataframe(nearby_fires)

        st.info(
            "**Why the variance?** The Master Stack uses 1km grid alignment. Original values are sampled from the raw files, while Master values are the result of Bilinear Interpolation (for weather/DEM) or Nearest Neighbor (for LULC). Small shifts are mathematically expected."
        )
else:
    with col_data:
        st.info("Click a point on the map to compare Master vs Original values.")
