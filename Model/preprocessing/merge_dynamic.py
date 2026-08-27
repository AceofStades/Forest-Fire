import os

# HDF5 takes file locks that ntfs-3g/FUSE mounts do not support; without this
# the netCDF reads below fail with an opaque "NetCDF: HDF error". Must be set
# before anything imports the HDF5 library.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import time

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from tqdm import tqdm

# --- 1. Paths ---
ERA5_NC_PATH = "dataset/ERA5-Land/era5_resampled_1km_v2.nc"
# Copernicus DEM GLO-30, averaged to the 1 km grid by dem_resample.py. The
# previous dem_resampled.tif came from CartoDEM (P5_PAN_CD_* tiles), an
# NRSC/ISRO product whose Bhuvan terms forbid redistributing a derivative --
# and whose 0-filled nodata holes produced artificial 1,861 m/km cliffs that
# fed straight into the Slope features.
DEM_PATH = "dataset/resampled-fix/dem_copernicus_1km.tif"
# ESA WorldCover 10 m, majority-aggregated to the 1 km grid by
# preprocessing/worldcover_resample.py. This replaces the earlier Bhuvan layer,
# which was reconstructed from a lossy RGB map render and which NRSC's terms do
# not permit us to redistribute. WorldCover is CC-BY-4.0 and carries real,
# named class codes.
LULC_PATH = "dataset/resampled-fix/worldcover_1km.tif"
GHS_PATH = "dataset/resampled-fix/ghs_resampled.tif"
MODIS_CSV_PATH = "dataset/MODIS/final-modis.csv"
OUTPUT_NC_PATH = "dataset/final_feature_stack_RELEASE_v2.nc"

# Minimum MODIS detection confidence (0-100) to accept as a fire label.
# 30 keeps the "nominal" and "high" classes and drops the noisiest detections.
MIN_CONFIDENCE = 30

# How long a cell stays alight in ACTIVE_FIRE after a detection. MODIS gets
# roughly four overpasses a day, so 12 h keeps a fire lit between plausible
# revisits without bridging every gap and collapsing back into a mask that
# never goes out. Raising this inflates the persistence baseline; the run
# prints that baseline so the cost is visible.
PERSISTENCE_HOURS = 12

# zlib level. Measured on this data: level 1 compresses the float32 weather
# fields to ratio 1.36 at 61 MB/s, level 5 to ratio 1.35 at 27 MB/s -- i.e.
# level 5 is 2.3x slower for no smaller file. Level 1 it is.
COMPLEVEL = 1

# Binary masks and small-code rasters do not need 32 bits. Storing the three
# fire channels as int8 cuts 2.2 GB of compression work to 0.55 GB, and int8
# masks compress ~13x faster per byte than float32 (796 vs 61 MB/s).
VAR_DTYPES = {
    "OBSERVED_FIRE": "int8",
    "ACTIVE_FIRE": "int8",
    "BURNED_AREA": "int8",
    "LULC": "uint8",
}


def load_and_clean_static(path, name, template_ds, interp_method="nearest"):
    """Loads a static GeoTIFF, cleans it, and aligns it to the template grid."""
    print(f"Processing {name} from {path}...")
    try:
        da = rioxarray.open_rasterio(path, masked=True).sel(band=1, drop=True)
        da = da.rename({"x": "longitude", "y": "latitude"})
        da.name = name

        # FIX: GHS_BUILT 255 -> 0
        if name == "GHS_BUILT":
            da = da.where(da != 255, 0)

        if "spatial_ref" in da.coords:
            da = da.drop_vars("spatial_ref")

        # Align to template
        da_aligned = da.interp_like(template_ds, method=interp_method)
        da_aligned = da_aligned.fillna(0)

        # Bilinear interpolation undershoots slightly around zero, which left a
        # few cells at -1e-16 in a field that is a percentage. Clamp to the
        # physical range so downstream sqrt/log transforms cannot produce NaN.
        if name == "GHS_BUILT":
            da_aligned = da_aligned.clip(0, 100)
        return da_aligned
    except Exception as e:
        print(f"Error loading {name}: {e}")
        raise


def nearest_index(grid, points):
    """Index of the nearest cell centre in `grid` for each value in `points`.

    `grid` must be monotonic (ascending or descending). np.searchsorted alone
    returns an *insertion* index, which always rounds to the same side and so
    displaces every point by half a cell on average (~500 m on the 1 km grid);
    here the two candidate neighbours are compared so the true nearest cell wins.

    Points outside the grid clamp to the nearest edge cell, matching the previous
    behaviour. Uses searchsorted rather than a full |grid - point| broadcast so
    the cost stays O(n log m) regardless of grid size.
    """
    ascending = grid[1] > grid[0]
    asc_grid = grid if ascending else grid[::-1]

    idx = np.searchsorted(asc_grid, points)
    idx = np.clip(idx, 1, len(asc_grid) - 1)

    left = asc_grid[idx - 1]
    right = asc_grid[idx]
    # Ties and out-of-range points resolve to the closer / nearest edge cell.
    idx = np.where(points - left < right - points, idx - 1, idx)

    if not ascending:
        idx = len(grid) - 1 - idx
    return idx


def generate_dynamic_fire_mask(csv_path, template_ds):
    """
    Generates a 3D (Time, Lat, Lon) binary fire mask from MODIS CSV points.
    """
    print(f"Generating Dynamic Fire Mask from {csv_path}...")

    # 1. Load CSV
    df = pd.read_csv(csv_path)
    # Parse timestamps
    # Assuming 'acq_timestamp' exists and is in a standard format
    # If not, construct from date and time
    if "acq_timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["acq_timestamp"])
    else:
        # Fallback: construct from acq_date + acq_time
        # acq_time is usually HHMM as integer. Need to pad.
        df["acq_time_str"] = df["acq_time"].astype(str).str.zfill(4)
        df["timestamp"] = pd.to_datetime(
            df["acq_date"] + " " + df["acq_time_str"], format="%Y-%m-%d %H%M"
        )

    # 2. Prepare Empty 3D Array matches ERA5 dimensions
    times = template_ds.valid_time.values
    lats = template_ds.latitude.values
    lons = template_ds.longitude.values

    fire_mask = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)

    # 3. Time Mapping
    print("Mapping fire events to nearest ERA5 hourly slot...")
    df["nearest_hour"] = df["timestamp"].dt.round("h")

    # Filter out points outside our time range
    # Ensure timezone naivety for comparison
    time_min = pd.to_datetime(times[0]).tz_localize(None)
    time_max = pd.to_datetime(times[-1]).tz_localize(None)

    # Ensure DF is also naive
    df["nearest_hour"] = df["nearest_hour"].dt.tz_localize(None)

    initial_count = len(df)
    df = df[(df["nearest_hour"] >= time_min) & (df["nearest_hour"] <= time_max)]
    filtered_count = len(df)
    print(f"  Points in time range: {filtered_count} / {initial_count}")

    # Discard detections outside the spatial domain. Clamping them to the nearest
    # index (the previous behaviour) would smear out-of-area fires onto the border
    # row/column and invent fire where none was observed.
    half_lat = abs(lats[1] - lats[0]) / 2
    half_lon = abs(lons[1] - lons[0]) / 2
    in_domain = (
        (df["latitude"] >= lats.min() - half_lat)
        & (df["latitude"] <= lats.max() + half_lat)
        & (df["longitude"] >= lons.min() - half_lon)
        & (df["longitude"] <= lons.max() + half_lon)
    )
    dropped = int((~in_domain).sum())
    df = df[in_domain]
    print(f"  Points in spatial domain: {len(df)} / {filtered_count} "
          f"({dropped} dropped as out-of-area)")

    # Confidence filter: MODIS reports 0-100. Low-confidence detections are
    # frequently false alarms, and an unfiltered label teaches the model noise.
    if "confidence" in df.columns and MIN_CONFIDENCE > 0:
        before = len(df)
        df = df[df["confidence"] >= MIN_CONFIDENCE]
        print(f"  Points with confidence >= {MIN_CONFIDENCE}: {len(df)} / {before}")

    # Keep only presumed vegetation fires (type 0); 1=volcano, 2=other static
    # land source, 3=offshore are not wildfire.
    if "type" in df.columns:
        before = len(df)
        df = df[df["type"] == 0]
        if len(df) != before:
            print(f"  Points of type 0 (vegetation fire): {len(df)} / {before}")

    # Create a lookup for time indices
    # Convert numpy datetime64 to pandas timestamp for robust lookup
    time_index_map = {
        pd.to_datetime(t).tz_localize(None): i for i, t in enumerate(times)
    }

    # Group by time step for efficiency
    grouped = df.groupby("nearest_hour")

    mapped_pixels = 0

    print("Rasterizing fire points...")
    for time_val, group in tqdm(grouped, desc="Processing Time Steps"):
        if time_val not in time_index_map:
            continue

        t_idx = time_index_map[time_val]

        # Get coordinates
        p_lats = group["latitude"].values
        p_lons = group["longitude"].values

        # Map to indices (nearest cell centre, see nearest_index)
        lat_idxs = nearest_index(lats, p_lats)
        lon_idxs = nearest_index(lons, p_lons)

        # Mark fire
        # We can use advanced indexing
        fire_mask[t_idx, lat_idxs, lon_idxs] = 1.0
        mapped_pixels += len(lat_idxs)

    print(f"Total fire pixels mapped: {mapped_pixels}")

    # --- Two honestly-named channels -------------------------------------
    #
    # OBSERVED_FIRE : exactly what MODIS saw, at the hour it saw it.
    # ACTIVE_FIRE   : a detection keeps its cell alight for PERSISTENCE_HOURS
    #                 and then goes out. The fill is strictly FORWARD in time,
    #                 so no frame is ever built from an observation that has not
    #                 happened yet.
    # BURNED_AREA   : running maximum of ACTIVE_FIRE, i.e. "has this cell burned
    #                 at any point up to now". Monotone by definition -- that is
    #                 what a burn scar is -- but named so nobody mistakes it for
    #                 an active-fire target.
    #
    # The superseded interpolate_fire_morphology.py did the opposite: on seeing a
    # detection at time t it back-filled the preceding 24 frames by dilating
    # toward frame t, so the "spread" it manufactured was a deterministic
    # function of the answer, and nothing ever extinguished.
    print(f"Applying causal persistence ({PERSISTENCE_HOURS} h)...")
    # int8 throughout: these are 0/1 masks, and it keeps peak RAM and the
    # compression bill down by 4x versus float32.
    observed = fire_mask.astype(np.int8)
    active = np.zeros_like(observed)
    t_idx_arr, y_idx_arr, x_idx_arr = np.nonzero(observed)
    for t_i, y_i, x_i in zip(t_idx_arr, y_idx_arr, x_idx_arr):
        active[t_i:t_i + PERSISTENCE_HOURS, y_i, x_i] = 1

    burned = np.maximum.accumulate(active, axis=0)
    fire_mask = observed

    coords = {"valid_time": times, "latitude": lats, "longitude": lons}
    dims = ("valid_time", "latitude", "longitude")
    fire_ds = xr.Dataset(
        {
            "OBSERVED_FIRE": xr.DataArray(fire_mask, coords=coords, dims=dims),
            "ACTIVE_FIRE": xr.DataArray(active, coords=coords, dims=dims),
            "BURNED_AREA": xr.DataArray(burned, coords=coords, dims=dims),
        }
    )

    obs_n, act_n, burn_n = fire_mask.sum(), active.sum(), burned.sum()
    print(f"  OBSERVED_FIRE pixel-hours: {obs_n:,.0f}")
    print(f"  ACTIVE_FIRE   pixel-hours: {act_n:,.0f} ({act_n / max(obs_n, 1):.1f}x observed)")
    print(f"  BURNED_AREA   pixel-hours: {burn_n:,.0f}")
    _report_persistence_baseline(active)

    return fire_ds


def _report_persistence_baseline(active):
    """Print the IoU a copy-the-input model would score, so it is on the record."""
    binary = active > 0.5
    print("  Persistence baseline on ACTIVE_FIRE (what copying the input scores):")
    for lead in (1, 8, 24):
        if lead >= binary.shape[0]:
            continue
        a, b = binary[:-lead], binary[lead:]
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        print(f"    lead {lead:>3} h: IoU = {inter / max(union, 1):.4f}")


VAR_METADATA = {
    "d2m": ("2 m dewpoint temperature", "K", "ERA5-Land"),
    "t2m": ("2 m air temperature", "K", "ERA5-Land"),
    "swvl1": ("Volumetric soil water, layer 1 (0-7 cm)", "m3 m-3", "ERA5-Land"),
    "e": ("Total evaporation", "m of water equivalent", "ERA5-Land"),
    "u10": ("10 m eastward wind component", "m s-1", "ERA5-Land"),
    "v10": ("10 m northward wind component", "m s-1", "ERA5-Land"),
    "tp": ("Total precipitation", "m", "ERA5-Land"),
    "cvl": ("Low vegetation cover fraction (static)", "1", "ERA5-Land"),
    "OBSERVED_FIRE": (
        "Binary mask of MODIS active-fire detections at their observed hour "
        "(no persistence, no interpolation)",
        "1", "MODIS (FIRMS) active fire product"),
    "ACTIVE_FIRE": (
        f"Binary active-fire mask; a detection keeps its cell alight for "
        f"{PERSISTENCE_HOURS} h and then extinguishes. Forward-fill only, so no "
        f"frame depends on a later observation",
        "1", "Derived from OBSERVED_FIRE"),
    "BURNED_AREA": (
        "Cumulative burn scar: 1 where the cell has been alight at any time up "
        "to and including this step. Monotone by construction",
        "1", "Derived from ACTIVE_FIRE"),
    "DEM": ("Terrain elevation above sea level", "m",
            "Copernicus DEM GLO-30 (2021), averaged per 1 km cell"),
    "LULC": ("Land cover class code; see worldcover_legend.csv", "1",
             "ESA WorldCover 10m v200 (2021), areal majority per 1 km cell"),
    "GHS_BUILT": ("Built-up surface share", "%", "GHSL GHS-BUILT-S R2023A (2018)"),
}


def add_metadata(ds):
    """Attach CF-style attributes so the published file is self-describing."""
    print("Attaching metadata...")
    for var, (long_name, units, source) in VAR_METADATA.items():
        if var in ds.data_vars:
            ds[var].attrs.update(
                {"long_name": long_name, "units": units, "source": source}
            )

    undocumented = [v for v in ds.data_vars if v not in VAR_METADATA]
    if undocumented:
        print(f"  WARNING: no metadata entry for {undocumented} -- these will ship "
              f"without units or a source. Add them to VAR_METADATA.")

    ds["latitude"].attrs.update(
        {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
    )
    ds["longitude"].attrs.update(
        {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
    )
    ds["valid_time"].attrs.update({"standard_name": "time", "axis": "T"})

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    ds.attrs.update(
        {
            "title": "Uttarakhand wildfire feature stack",
            "summary": (
                "Hourly co-registered weather, terrain, land cover and MODIS "
                "active-fire labels on a ~1 km grid over Uttarakhand, India."
            ),
            "institution": "Forest-Fire project",
            "source": ("ERA5-Land, MODIS active fire, Copernicus DEM GLO-30, "
                       "ESA WorldCover, GHSL"),
            "Conventions": "CF-1.8",
            "geospatial_lat_min": float(lats.min()),
            "geospatial_lat_max": float(lats.max()),
            "geospatial_lon_min": float(lons.min()),
            "geospatial_lon_max": float(lons.max()),
            "geospatial_lat_resolution": float(abs(lats[1] - lats[0])),
            "geospatial_lon_resolution": float(abs(lons[1] - lons[0])),
            "spatial_ref": "EPSG:4326",
            "time_coverage_start": str(ds["valid_time"].values[0]),
            "time_coverage_end": str(ds["valid_time"].values[-1]),
            "modis_min_confidence": MIN_CONFIDENCE,
            "coordinate_convention": "Coordinates are cell centres.",
        }
    )
    return ds


def main():
    print(f"--- Starting Dynamic Merge (Output: {OUTPUT_NC_PATH}) ---")

    # 1. Load ERA5 Template (Time Series)
    #
    # Read straight into RAM rather than through dask. The file is stored with
    # chunks of 366 along valid_time; opening it with chunks={"valid_time": 50}
    # made every dask block decompress a whole 366-step stored chunk, so the
    # source was inflated roughly sevenfold on the way through. The full array
    # is ~5.8 GB, which is cheaper to just hold.
    print(f"Loading ERA5 from {ERA5_NC_PATH} into RAM...")
    t_load = time.time()
    # engine="netcdf4" for BOTH read and write. netCDF4 and h5netcdf each bring
    # their own HDF5 library; mixing them in one process makes the HDF5
    # dimension-scale calls fail ("Unspecified error in H5DSget_num_scales").
    era5_ds = xr.open_dataset(ERA5_NC_PATH, engine="netcdf4").load()
    if "spatial_ref" in era5_ds.coords:
        era5_ds = era5_ds.drop_vars("spatial_ref")
    print(f"  loaded {sum(era5_ds[v].nbytes for v in era5_ds.data_vars) / 1e9:.2f} GB "
          f"in {time.time() - t_load:.1f}s")

    # 2. Load Static Layers
    dem = load_and_clean_static(DEM_PATH, "DEM", era5_ds, "linear")
    lulc = load_and_clean_static(LULC_PATH, "LULC", era5_ds, "nearest")
    ghs = load_and_clean_static(GHS_PATH, "GHS_BUILT", era5_ds, "linear")

    # 3. Generate Fire Channels (OBSERVED_FIRE / ACTIVE_FIRE / BURNED_AREA)
    fire_ds = generate_dynamic_fire_mask(MODIS_CSV_PATH, era5_ds)

    # 4. Crop to Valid Data Area (Using DEM)
    print("Calculating Valid Data Crop...")
    # DEM is 2D (Lat, Lon), fire is 3D (Time, Lat, Lon).
    # We use DEM to determine spatial bounds.
    valid_mask = (dem > 0) & (dem.notnull())
    valid_coords = np.argwhere(valid_mask.values)

    if len(valid_coords) == 0:
        print("Error: No valid data found in DEM. Check source files.")
        return

    lat_min_idx = valid_coords[:, 0].min()
    lat_max_idx = valid_coords[:, 0].max()
    lon_min_idx = valid_coords[:, 1].min()
    lon_max_idx = valid_coords[:, 1].max()

    print(
        f"  Cropping Lat Index: {lat_min_idx}-{lat_max_idx}, Lon Index: {lon_min_idx}-{lon_max_idx}"
    )

    # Apply Crop. Slices are half-open, so the +1 is what keeps the last valid
    # row and column found above inside the output instead of trimming them.
    def crop(da):
        return da.isel(
            latitude=slice(lat_min_idx, lat_max_idx + 1),
            longitude=slice(lon_min_idx, lon_max_idx + 1),
        )

    print("Cropping datasets...")
    era5_cropped = crop(era5_ds)
    dem_cropped = crop(dem)
    lulc_cropped = crop(lulc)
    ghs_cropped = crop(ghs)
    fire_cropped = crop(fire_ds)

    # 5. Merge
    print("Merging datasets...")

    # Create the final Dataset
    # ERA5 is the base. We add static layers as variables (coordinates?)
    # Usually static layers are just variables with (lat, lon) dims, not (time, lat, lon).
    # Xarray handles this fine.

    # However, 'fire_cropped' IS 3D (time, lat, lon).

    final_ds = xr.merge(
        [
            era5_cropped,
            fire_cropped,
            dem_cropped.astype(np.float32),
            lulc_cropped.astype(np.int16),
            ghs_cropped.astype(np.float32),
        ]
    )

    # Collapse variables that never change over time to 2D. 'cvl' in particular
    # is a static vegetation-cover field that ERA5 carried along a 1464-step time
    # axis, costing ~716 MB for 1464 identical copies.
    n_steps = final_ds.sizes["valid_time"]
    for var in list(final_ds.data_vars):
        da = final_ds[var]
        if "valid_time" not in da.dims or n_steps < 2:
            continue

        # Cheap rejection first: a handful of probe frames rules out almost every
        # time-varying field without materialising the whole ~716 MB array.
        probes = sorted({0, n_steps // 3, 2 * n_steps // 3, n_steps - 1})
        first = np.nan_to_num(da.isel(valid_time=probes[0]).values)
        if not all(
            np.array_equal(np.nan_to_num(da.isel(valid_time=p).values), first)
            for p in probes[1:]
        ):
            continue

        # Probes agreed, so confirm against every step before collapsing.
        values = da.values
        if np.all(np.nan_to_num(values) == np.nan_to_num(values[0])):
            print(f"  '{var}' is constant over time -> storing as 2D")
            final_ds[var] = da.isel(valid_time=0, drop=True)

    # Force Dimension Order
    final_ds = final_ds.transpose("valid_time", "latitude", "longitude")

    final_ds = add_metadata(final_ds)

    # 6. Save
    print(f"Saving to {OUTPUT_NC_PATH}...")

    # Compression. Chunks are sized to the stored layout we want on read
    # (a whole spatial field per chunk, a slab of time) so that neither this
    # write nor a later dask read has to decompress more than it needs.
    # Variables carry encoding inherited from the source file, including its
    # chunksizes. 'cvl' was just collapsed from 3D to 2D, so the inherited
    # rank-3 chunk spec no longer matches its rank and HDF5 rejects the write.
    # Clear it all and state every setting explicitly.
    for var in final_ds.data_vars:
        final_ds[var].encoding = {}
        # rioxarray leaves identity scale_factor/add_offset attributes behind.
        # They do nothing numerically, but CF decoding sees them and promotes
        # the variable to float64 on read -- so an int8 mask or a uint8 class
        # code comes back as 8-byte floats for every user of the dataset.
        for attr in ("scale_factor", "add_offset"):
            val = final_ds[var].attrs.get(attr)
            if val is not None and float(val) == (1.0 if attr == "scale_factor" else 0.0):
                del final_ds[var].attrs[attr]
    for coord in final_ds.coords:
        final_ds[coord].encoding = {
            k: v for k, v in final_ds[coord].encoding.items()
            if k in ("units", "calendar", "dtype")
        }

    encoding = {}
    for var in final_ds.data_vars:
        # shuffle reorders bytes before deflate; on float32 fields it buys a
        # noticeably better ratio for almost no CPU, and it is inert for int8.
        enc = {"zlib": True, "complevel": COMPLEVEL, "shuffle": True}
        if var in VAR_DTYPES:
            enc["dtype"] = VAR_DTYPES[var]
        da = final_ds[var]
        if "valid_time" in da.dims:
            enc["chunksizes"] = (
                min(366, da.sizes["valid_time"]),
                da.sizes["latitude"],
                da.sizes["longitude"],
            )
        encoding[var] = enc

    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_NC_PATH), exist_ok=True)

    if os.path.exists(OUTPUT_NC_PATH):
        os.remove(OUTPUT_NC_PATH)

    raw_gb = sum(final_ds[v].nbytes for v in final_ds.data_vars) / 1e9
    print(f"  {raw_gb:.2f} GB uncompressed, zlib level {COMPLEVEL}")
    t_write = time.time()
    final_ds.to_netcdf(OUTPUT_NC_PATH, format="netcdf4", engine="netcdf4",
                       encoding=encoding)
    dt = time.time() - t_write
    size_gb = os.path.getsize(OUTPUT_NC_PATH) / 1e9
    print(f"  wrote {size_gb:.2f} GB in {dt:.1f}s "
          f"({raw_gb / dt * 1000:.0f} MB/s, ratio {raw_gb / size_gb:.2f}x)")

    print("✅ Done! Dynamic dataset created.")


if __name__ == "__main__":
    main()
