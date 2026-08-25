"""
Follow-up audit: where does every fire pixel in the published file come from?

HISTORICAL. This script audits the OLD artifact,
final_feature_stack_DYNAMIC_interpolated.nc, and is kept as the record of why
that file was withdrawn. The corrected pipeline writes
final_feature_stack_RELEASE.nc with OBSERVED_FIRE / ACTIVE_FIRE / BURNED_AREA;
audit that one with audit_release_readiness.py instead.

Compares the raw MODIS detections -> final_feature_stack_DYNAMIC_new.nc (rasterised)
-> final_feature_stack_DYNAMIC_interpolated.nc (published), and quantifies how much
of the published label is observation and how much is synthesised.

Run from the Model/ directory:
    ../.venv/bin/python dataset-validation-scripts/audit_fire_label_provenance.py
"""

import os

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

PUB = "dataset/final_feature_stack_DYNAMIC_interpolated.nc"
RAW = "dataset/final_feature_stack_DYNAMIC_new.nc"
MODIS_CSV = "dataset/MODIS/final-modis.csv"
LULC_TIF = "dataset/resampled-fix/lulc_resampled.tif"
LULC_SRC = "dataset/LULC/UK_LULC50K_2016.tif"
ERA5_1KM = "dataset/ERA5-Land/era5_resampled_1km.nc"
ERA5_SRC = "dataset/ERA5-Land/final-era5_rechunked.nc"


def hdr(n, t):
    print(f"\n{'=' * 72}\n[{n}] {t}\n{'=' * 72}")


def observed_vs_synthetic():
    hdr(1, "How much of the published fire label is actually observed?")
    df = pd.read_csv(MODIS_CSV)
    n_det = len(df)

    pub = xr.open_dataset(PUB, engine="h5netcdf")["MODIS_FIRE_T1"].values > 0.5
    print(f"  raw MODIS detections in csv          : {n_det:,}")

    if os.path.exists(RAW):
        raw = xr.open_dataset(RAW, engine="h5netcdf")["MODIS_FIRE_T1"].values > 0.5
        print(f"  rasterised fire pixel-hours (_new.nc): {raw.sum():,}")
    else:
        raw = None
        print("  _new.nc not found; skipping rasterised comparison")

    print(f"  published fire pixel-hours (_interp) : {pub.sum():,}")

    ever = pub.any(axis=0)
    print(f"  distinct cells that ever burn        : {ever.sum():,}")
    print(f"  mean burn duration per burning cell  : {pub.sum() / max(ever.sum(), 1):.1f} hours "
          f"({pub.sum() / max(ever.sum(), 1) / 24:.1f} days)")

    if raw is not None:
        synth = pub.sum() - raw.sum()
        print(f"\n  -> synthesised pixel-hours: {synth:,} "
              f"({100 * synth / max(pub.sum(), 1):.2f}% of the published label)")
        # are all observed pixels preserved?
        lost = np.logical_and(raw, ~pub).sum()
        print(f"  -> observed pixel-hours dropped by interpolation: {lost:,}")

    print(f"\n  inflation factor vs raw detections: {pub.sum() / n_det:.1f}x")
    return pub


def leakage_from_future(pub):
    hdr(2, "Is the synthesised growth derived from the FUTURE observation?")
    print("  interpolate_fire_morphology.py, on detecting a jump at time t, writes")
    print("  frames start_t+1 .. t by dilating the previous frame toward current_frame[t].")
    print("  Those intermediate frames are therefore a deterministic function of the")
    print("  frame a model is being asked to predict.\n")

    per_frame = pub.sum(axis=(1, 2))
    d = np.diff(per_frame.astype(int))
    jumps = np.where(d > 0)[0] + 1
    print(f"  frames where the mask grows: {len(jumps)}")

    # For each growth event, check whether growth is a nested dilation (each step's
    # new fire is a superset of the previous) -- the signature of EDT back-fill.
    nested = 0
    checked = 0
    for t in jumps[:200]:
        if t < 1:
            continue
        a, b = pub[t - 1], pub[t]
        if np.logical_and(a, ~b).sum() == 0 and np.logical_and(b, ~a).sum() > 0:
            nested += 1
        checked += 1
    print(f"  growth steps that are strictly nested (old fire never turns off): "
          f"{nested}/{checked}")
    if checked and nested / checked > 0.9:
        print("  -> Confirms monotone dilation. The mask encodes the endpoint shape,")
        print("     so 'predicting spread' reduces to inverting a known geometric rule.")

    # Distance-transform determinism: growth rings should be equidistant from the seed
    print("\n  Sanity: fraction of frames identical to the previous frame")
    same = (np.diff(per_frame.astype(int)) == 0).mean()
    identical = sum(np.array_equal(pub[t - 1], pub[t]) for t in range(1, 400)) / 399
    print(f"  same total: {same:.3f}   byte-identical (first 400 frames): {identical:.3f}")


def lulc_semantics():
    hdr(3, "LULC: do the stored class codes mean anything?")
    ds = xr.open_dataset(PUB, engine="h5netcdf")
    lulc = ds["LULC"].values
    vals, cnts = np.unique(lulc, return_counts=True)
    order = np.argsort(-cnts)
    print(f"  distinct codes in published file: {len(vals)}")
    print("  top 15 codes by area:")
    for i in order[:15]:
        print(f"    code {int(vals[i]):>4}: {cnts[i]:>8,} px ({100 * cnts[i] / lulc.size:5.2f}%)")

    if os.path.exists(LULC_SRC):
        with rasterio.open(LULC_SRC) as src:
            print(f"\n  source raster: {LULC_SRC}")
            print(f"    dtype={src.dtypes[0]} count={src.count} nodata={src.nodata}")
            print(f"    crs={src.crs} size={src.width}x{src.height}")
            try:
                cmap = src.colormap(1)
                print(f"    HAS A COLOR TABLE with {len(cmap)} entries")
                print("    -> the band holds palette indices, not thematic class codes")
            except ValueError:
                print("    no color table (band values are the classes)")

    print("\n  src/dataset.py:51 builds Water_Mask as (LULC > 0), i.e. it assumes")
    print("  class 0 == water/non-burnable. Check that against the codes above.")
    zero_frac = (lulc == 0).mean()
    print(f"  fraction of the map with LULC == 0: {zero_frac:.4f}")


def _best_shift(obs, ref, rng=9, margin=12):
    """Return (rmse, dy, dx) of the roll that best aligns obs onto ref."""
    best = None
    m = slice(margin, -margin)
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            o = np.roll(np.roll(obs, dy, axis=0), dx, axis=1)
            r = np.sqrt(np.nanmean((o[m, m] - ref[m, m]) ** 2))
            if best is None or r < best[0]:
                best = (r, dy, dx)
    return best


def era5_halfpixel():
    hdr(4, "ERA5 resampling: half-pixel origin error (measured, not assumed)")
    if not os.path.exists(ERA5_SRC):
        print(f"  {ERA5_SRC} not found; skipping.")
        return
    src = xr.open_dataset(ERA5_SRC, engine="h5netcdf")
    lat, lon = src.latitude.values, src.longitude.values
    yres, xres = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    print(f"  source ERA5-Land grid: {len(lat)} x {len(lon)} @ {yres:.2f} deg")
    print(f"  lat[0]={lat[0]:.2f} lon[0]={lon[0]:.2f} -- these are grid points (cell CENTERS)")
    print("\n  era5-resample.py:53 passes them to rasterio.transform.from_origin(),")
    print("  which expects the OUTER CORNER of the top-left cell.")
    print(f"  Predicted error: half a source cell = {yres / 2:.3f} deg = "
          f"{yres / 2 * 111:.2f} km = {yres / 2 / 0.009:.1f} output pixels.\n")

    pub = xr.open_dataset(PUB, engine="h5netcdf")
    plat, plon = pub.latitude.values, pub.longitude.values
    obs = pub["t2m"].isel(valid_time=0).values
    ref = (src["t2m"].isel(valid_time=0)
           .interp(latitude=("latitude", plat), longitude=("longitude", plon),
                   method="linear").values)
    rmse0 = np.sqrt(np.nanmean((obs[12:-12, 12:-12] - ref[12:-12, 12:-12]) ** 2))
    r, dy, dx = _best_shift(obs, ref)
    cell_km = 0.009 * 111
    print(f"  measured: t2m RMSE vs correct regrid = {rmse0:.4f} K as stored")
    print(f"            drops to {r:.4f} K when rolled by dy={dy:+d}, dx={dx:+d}")
    print(f"  -> weather is displaced {abs(dy) * cell_km:.2f} km N-S and "
          f"{abs(dx) * cell_km:.2f} km E-W (predicted {yres / 2 * 111:.2f} km)")

    # Terrain went through static_sampler.py, which builds its transform correctly.
    if os.path.exists("dataset/DEM/merged_dem.tif"):
        import rasterio
        from rasterio.warp import Resampling, reproject
        dlat, dlon = abs(plat[1] - plat[0]), abs(plon[1] - plon[0])
        tr = rasterio.transform.from_origin(plon[0] - dlon / 2, plat[0] + dlat / 2,
                                            dlon, dlat)
        dst = np.empty((len(plat), len(plon)), dtype=np.float32)
        with rasterio.open("dataset/DEM/merged_dem.tif") as s:
            reproject(source=rasterio.band(s, 1), destination=dst, dst_transform=tr,
                      dst_crs="EPSG:4326",
                      src_nodata=s.nodata if s.nodata is not None else 0, dst_nodata=0,
                      resampling=Resampling.bilinear)
        dr, ddy, ddx = _best_shift(pub["DEM"].values, dst, rng=6, margin=10)
        print(f"\n  DEM, by contrast, aligns best at dy={ddy:+d}, dx={ddx:+d} "
              f"(RMSE {dr:.1f} m)")
        if (ddy, ddx) == (0, 0) and (dy, dx) != (0, 0):
            print("  -> Terrain is correctly placed; the weather channels are not.")
            print("     The dataset is internally inconsistent: every weather value is")
            print("     read from a location ~5 km away from the terrain and fire label")
            print("     it is paired with.")


def main():
    if not os.path.exists(PUB):
        print(f"Not found: {PUB}. Run from the Model/ directory.")
        return
    pub = observed_vs_synthetic()
    leakage_from_future(pub)
    lulc_semantics()
    era5_halfpixel()
    print(f"\n{'=' * 72}\nDone.\n{'=' * 72}")


if __name__ == "__main__":
    main()
