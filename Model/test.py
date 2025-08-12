import xarray as xr

cvl = xr.open_dataset("dataset/ERA5-Land/era5-april/cvl.area-subset.31.5.81.1.28.7.77.5.nc")
era_april = xr.open_dataset("dataset/ERA5-Land/era5-april/data_0.nc")
era_may = xr.open_dataset("dataset/ERA5-Land/era5-may/data_0.nc")

cvl_static = cvl.isel(time=0).drop_vars("time")

cvl_static_reindexed = cvl_static.reindex_like(era_april, method='nearest')

era_april_merged = era_april.merge(cvl_static_reindexed)
era_may_merged = era_may.merge(cvl_static_reindexed)

final_era_dataset = xr.concat([era_april_merged, era_may_merged], dim='time')

final_era_dataset.to_netcdf("merged_era5.nc", format='NETCDF4')

print("Final merged NetCDF file has been saved as 'merged_era5.nc'.")
