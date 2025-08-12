import streamlit as st
import xarray as xr
import pandas as pd

st.set_page_config(layout="wide")

ds1 = xr.open_dataset("dataset/ERA5-Land/era5-april/cvl.area-subset.31.5.81.1.28.7.77.5.nc")
df1 = ds1.to_dataframe()

ds2 = xr.open_dataset("dataset/ERA5-Land/era5-april/data_0.nc")
df2 = ds2.to_dataframe()

ds3 = xr.open_dataset("dataset/ERA5-Land/era5-may/data_0.nc")
df3 = ds3.to_dataframe()

df4 = pd.read_csv("dataset/MODIS/modis_2016_India.csv")

ds5 = xr.open_dataset("dataset/ERA5-Land/final-era5.nc")
df5 = ds5.to_dataframe()

df6 = pd.read_csv("dataset/MODIS/final-modis.csv")

st.subheader("CVL")
st.dataframe(df1.head())
st.subheader("ERA5-April")
st.dataframe(df2.head())
st.subheader("ERA5-May")
st.dataframe(df3.head())
st.subheader("MODIS")
st.dataframe(df4)
st.subheader("Final-ERA5")
st.dataframe(df5)
st.subheader("Final-MODIS")
st.dataframe(df6)
