import xarray as xr
ds = xr.open_dataset("dataset/final_feature_stack_DYNAMIC_new.nc", engine="h5netcdf")
print(ds["MODIS_FIRE_T1"].shape)
burn_scar = ds["MODIS_FIRE_T1"].cumsum(dim="valid_time")
print(burn_scar.max().item())
