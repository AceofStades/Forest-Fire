import xarray as xr
import numpy as np

path = "Model/dataset/final_feature_stack_DYNAMIC_interpolated.nc"
try:
    ds = xr.open_dataset(path, engine="h5netcdf")
    print("Dimensions:", ds.dims)
    print("Variables:", list(ds.data_vars.keys()))
    fire = ds["MODIS_FIRE_T1"].values
    print("Fire shape:", fire.shape)
    print("Fire max:", np.nanmax(fire))
    print("Total fire pixels:", np.nansum(fire))
    print("Non-zero fire frames:", (fire.sum(axis=(1,2)) > 0).sum())
except Exception as e:
    print(e)
