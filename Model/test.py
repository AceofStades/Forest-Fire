import xarray as xr

cvl = xr.open_dataset("dataset/ERA5-Land/era5-april/cvl.area-subset.31.5.81.1.28.7.77.5.nc")
era_april = xr.open_dataset("dataset/ERA5-Land/era5-april/data_0.nc")
era_may = xr.open_dataset("dataset/ERA5-Land/era5-may/data_0.nc")

cvl = cvl.drop_dims("time")

print(cvl)
cvl.to_netcdf("test_cvl.nc")
print("saved")
# print("\n\n\n\n")
# print(era_april)
# print("\n\n\n\n")
# print(era_may)
