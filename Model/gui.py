import streamlit as st
import xarray as xr
import pandas as pd

st.set_page_config(layout="wide")

ds1 = xr.open_dataset("dataset/ERA5-Land/era5-april/cvl.area-subset.31.5.81.1.28.7.77.5.nc")
df1 = ds1.to_dataframe()

ds2 = xr.open_dataset("dataset/ERA5-Land/era5-april/data_0.nc")
df2 = ds2.to_dataframe()

ds3 = xr.open_dataset("dataset/ERA5-Land/era5-may/cvl.area-subset.31.5.81.1.28.7.77.5.nc")
df3 = ds3.to_dataframe()

ds4 = xr.open_dataset("dataset/ERA5-Land/era5-may/data_0.nc")
df4 = ds4.to_dataframe()

df5 = pd.read_csv("dataset/MODIS/modis_2016_India.csv")

ds6 = xr.open_dataset("dataset/ERA5-Land/merged_era5.nc")
df6 = ds6.to_dataframe()

st.dataframe(df1.head())
st.dataframe(df2.head())
st.dataframe(df3.head())
st.dataframe(df4.head())
# st.dataframe(df5.head())
st.dataframe(df6)
