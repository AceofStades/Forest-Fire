import os

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

st.set_page_config(layout="wide")

FINAL_MASTER_PATH = "dataset/final_feature_stack_MASTER.nc"
MODIS_OG = "dataset/MODIS/modis_2016_India.csv"
MODIS_FINAL = "dataset/MODIS/final-modis.csv"
ERA5_PATH = "dataset/ERA5-Land/era5-april/data_0.nc"
ERA5_PATH_RE = "dataset/ERA5-Land/final-era5_rechunked.nc"


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

            if var_name == "MODIS_FIRE_T1":
                search_data = data_var.compute()
            else:
                search_data = data_var.isel(valid_time=0).compute()

            raw_values = search_data.values
            indices = np.argwhere(raw_values > 0)

            if len(indices) == 0:
                return [
                    pd.DataFrame({"Info": [f"No non-zero values found for {var_name}"]})
                ]

            valid_clusters = []
            for idx in indices:
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


st.title("Forest Fire Project Diagnostics")
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

if os.path.exists("dataset/DEM/dem_plot.png"):
    st.image("dataset/DEM/dem_plot.png", caption="Elevation")
if os.path.exists("dataset/GHS/ghs_downsampled_plot.png"):
    st.image("dataset/GHS/ghs_downsampled_plot.png", caption="Settlements")
if os.path.exists("dataset/LULC/lulc.png"):
    st.image("dataset/LULC/lulc.png", caption="Land Use")

st.subheader("Original ERA5-Land")
st.dataframe(get_data_safe(ERA5_PATH))

st.subheader("Final ERA5-Land (Rechunked)")
st.dataframe(get_data_safe(ERA5_PATH_RE))
