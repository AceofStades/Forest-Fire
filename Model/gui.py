import streamlit as st
import xarray as xr
# from netCDF4 import Dataset

# file = Dataset("dataset/data_stream-oper_stepType-accum.nc")
ds1 = xr.open_dataset("dataset/data_stream-oper_stepType-accum.nc")
df1 = ds1.to_dataframe()

ds2 = xr.open_dataset("dataset/data_stream-oper_stepType-instant.nc")
df2 = ds2.to_dataframe()

vbds1 = xr.open_dataset("vishal-datasets/merged_era5_2015_2016.nc")
vbdf1 = vbds1.to_dataframe()

vbds2 = xr.open_dataset("vishal-datasets/viirs_binary_fire_2015_2016.nc")
vbdf2 = vbds2.to_dataframe()

# print(df.head())
# st.dataframe(df1.head())
# st.dataframe(df2.head())

# st.dataframe(vbdf1.head())
# st.dataframe(vbdf2.head())

st.dataframe(df2)
