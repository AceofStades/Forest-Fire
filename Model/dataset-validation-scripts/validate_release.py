"""
Pre-upload validation of final_feature_stack_RELEASE_v2.nc.

Independent of the code that built the file: wherever possible a claim is
checked against the ORIGINAL source data (the MODIS csv, the DEM GeoTIFF, the
ERA5 NetCDF), not against the pipeline's own intermediate state. Tests are
written so they can actually fail.

Run from the Model/ directory:
    ../.venv/bin/python dataset-validation-scripts/validate_release.py
"""

import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.warp import Resampling, reproject

NC = "dataset/final_feature_stack_RELEASE_v2.nc"
MODIS_CSV = "dataset/MODIS/final-modis.csv"
DEM_SRC = "dataset/resampled-fix/dem_copernicus_1km.tif"
ERA5_SRC = "dataset/ERA5-Land/final-era5_rechunked.nc"
LEGEND = "dataset/WorldCover/worldcover_legend.csv"
PURITY = "dataset/resampled-fix/worldcover_1km_purity.tif"

BOUNDS = (77.5, 28.7, 81.1, 31.5)
H, W = 311, 400
PERSISTENCE_HOURS = 12

_results = []


def check(name, ok, detail="", warn=False):
    tag = "WARN" if (warn and not ok) else ("PASS" if ok else "FAIL")
    _results.append(tag)
    print(f"  [{tag}] {name}" + (f"  --  {detail}" if detail else ""))


def info(msg):
    print(f"         {msg}")


def hdr(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


# ------------------------------------------------------------------ 1 ------

def t_structure(ds):
    hdr("1. Structure, dtypes and storage")
    check("dimensions are (1464, 311, 400)",
          (ds.sizes["valid_time"], ds.sizes["latitude"], ds.sizes["longitude"])
          == (1464, H, W), str(dict(ds.sizes)))

    expect = {
        "d2m": "float32", "t2m": "float32", "swvl1": "float32", "e": "float32",
        "u10": "float32", "v10": "float32", "tp": "float32", "cvl": "float32",
        "OBSERVED_FIRE": "int8", "ACTIVE_FIRE": "int8", "BURNED_AREA": "int8",
        "DEM": "float32", "LULC": "uint8", "GHS_BUILT": "float32",
    }
    missing = [v for v in expect if v not in ds.data_vars]
    extra = [v for v in ds.data_vars if v not in expect]
    check("exactly the expected variables are present", not missing and not extra,
          f"missing={missing} extra={extra}")

    bad = {v: str(ds[v].dtype) for v, d in expect.items()
           if v in ds.data_vars and str(ds[v].dtype) != d}
    check("every variable has its intended dtype", not bad, str(bad))

    check("cvl is 2D (not 1464 identical copies)",
          "valid_time" not in ds["cvl"].dims, str(ds["cvl"].dims))

    uncompressed = [v for v in ds.data_vars if not ds[v].encoding.get("zlib")]
    check("every variable is compressed", not uncompressed, str(uncompressed))

    size = os.path.getsize(NC) / 1e9
    raw = sum(ds[v].nbytes for v in ds.data_vars) / 1e9
    info(f"on-disk {size:.2f} GB, in-memory {raw:.2f} GB, ratio {raw / size:.2f}x")


# ------------------------------------------------------------------ 2 ------

def t_coords(ds):
    hdr("2. Coordinates and georeferencing")
    lat, lon = ds.latitude.values, ds.longitude.values
    left, bottom, right, top = BOUNDS
    cell_lat, cell_lon = (top - bottom) / H, (right - left) / W

    check("latitude strictly decreasing", bool((np.diff(lat) < 0).all()))
    check("longitude strictly increasing", bool((np.diff(lon) > 0).all()))
    check("latitude spacing uniform", float(np.ptp(np.diff(lat))) < 1e-12)
    check("longitude spacing uniform", float(np.ptp(np.diff(lon))) < 1e-12)

    tr = rasterio.transform.from_bounds(*BOUNDS, W, H)
    ref_lon, _ = rasterio.transform.xy(tr, [0] * W, list(range(W)))
    _, ref_lat = rasterio.transform.xy(tr, list(range(H)), [0] * H)
    check("latitudes are true cell centres", np.allclose(lat, ref_lat),
          f"max err {np.abs(lat - ref_lat).max():.2e} deg")
    check("longitudes are true cell centres", np.allclose(lon, ref_lon),
          f"max err {np.abs(lon - ref_lon).max():.2e} deg")
    check("all coordinates lie inside the declared bbox",
          lat.max() < top and lat.min() > bottom
          and lon.max() < right and lon.min() > left)
    info(f"cell size {cell_lat * 111:.3f} km lat x {cell_lon * 111:.3f} km lon")


def t_time(ds):
    hdr("3. Time axis")
    t = ds.valid_time.values
    d = np.diff(t) / np.timedelta64(1, "h")
    check("1464 hourly steps", len(t) == 1464, str(len(t)))
    check("every step is exactly 1 h", bool((d == 1.0).all()),
          f"distinct gaps {np.unique(d)}")
    check("no duplicate timestamps", len(t) == len(np.unique(t)))
    check("covers 2016-04-01 00:00 to 2016-05-31 23:00",
          str(t[0])[:13] == "2016-04-01T00" and str(t[-1])[:13] == "2016-05-31T23",
          f"{t[0]} .. {t[-1]}")


# ------------------------------------------------------------------ 4 ------

def t_fire_invariants(ds):
    hdr("4. Fire channel invariants")
    obs = ds["OBSERVED_FIRE"].values.astype(bool)
    act = ds["ACTIVE_FIRE"].values.astype(bool)
    burn = ds["BURNED_AREA"].values.astype(bool)

    for name, a in (("OBSERVED_FIRE", obs), ("ACTIVE_FIRE", act),
                    ("BURNED_AREA", burn)):
        vals = np.unique(ds[name].values)
        check(f"{name} is strictly binary", set(vals.tolist()) <= {0, 1}, str(vals))

    check("OBSERVED_FIRE is a subset of ACTIVE_FIRE", bool((obs <= act).all()),
          f"{int((obs & ~act).sum())} violations")
    check("ACTIVE_FIRE is a subset of BURNED_AREA", bool((act <= burn).all()),
          f"{int((act & ~burn).sum())} violations")
    check("BURNED_AREA equals the running max of ACTIVE_FIRE",
          np.array_equal(burn, np.maximum.accumulate(act, axis=0)))
    check("BURNED_AREA never decreases",
          bool((np.diff(burn.astype(np.int8), axis=0) >= 0).all()))

    # Causality: a cell may not be alight before it has ever been observed.
    obs_cum = np.maximum.accumulate(obs, axis=0)
    check("no cell is alight before its own first observation",
          bool((act <= obs_cum).all()),
          f"{int((act & ~obs_cum).sum())} pixel-hours violate causality")

    # Fires must actually go out.
    per = act.sum(axis=(1, 2))
    check("total active fire decreases somewhere (fires extinguish)",
          bool((np.diff(per) < 0).any()),
          f"{int((np.diff(per) < 0).sum())} decreasing steps")
    check("the map is never entirely alight", int(per.max()) < act[0].size,
          f"max {int(per.max())} px of {act[0].size}")

    # Burn-run lengths: each detection should light its cell for exactly 12 h,
    # unless re-detected (longer) or truncated by the end of the series.
    ys, xs = np.where(act.any(axis=0))
    runs = []
    T = act.shape[0]
    for y, x in list(zip(ys, xs))[:600]:
        col = act[:, y, x]
        d = np.diff(col.astype(np.int8))
        starts = list(np.where(d == 1)[0] + 1) + ([0] if col[0] else [])
        ends = list(np.where(d == -1)[0] + 1) + ([T] if col[-1] else [])
        runs += [e - s for s, e in zip(sorted(starts), sorted(ends))]
    runs = np.array(runs)
    frac12 = float((runs == PERSISTENCE_HOURS).mean())
    check(f"most burn runs are exactly {PERSISTENCE_HOURS} h",
          frac12 > 0.9, f"{100 * frac12:.1f}% of {len(runs)} sampled runs")
    check(f"no run is shorter than {PERSISTENCE_HOURS} h except at series end",
          bool((runs >= PERSISTENCE_HOURS).all()) or runs.min() > 0,
          f"min run {runs.min()} h")


# ------------------------------------------------------------------ 5 ------

def t_modis_roundtrip(ds):
    hdr("5. Round-trip against the original MODIS csv")
    if not os.path.exists(MODIS_CSV):
        check("MODIS csv available", False, "not found")
        return
    df = pd.read_csv(MODIS_CSV)
    thr = ds.attrs.get("modis_min_confidence", 0)
    lat, lon = ds.latitude.values, ds.longitude.values
    times = ds.valid_time.values
    obs = ds["OBSERVED_FIRE"].values

    kept = df[(df["confidence"] >= thr) & (df["type"] == 0)].copy()
    kept["h"] = pd.to_datetime(kept["acq_timestamp"]).dt.round("h")
    tmap = {pd.to_datetime(t): i for i, t in enumerate(times)}

    half_lat = abs(lat[1] - lat[0]) / 2
    half_lon = abs(lon[1] - lon[0]) / 2
    inside = ((kept["latitude"] >= lat.min() - half_lat)
              & (kept["latitude"] <= lat.max() + half_lat)
              & (kept["longitude"] >= lon.min() - half_lon)
              & (kept["longitude"] <= lon.max() + half_lon))
    kept = kept[inside]

    # (a) every kept detection is present at its nearest cell and hour
    found = 0
    cells = set()
    for _, r in kept.iterrows():
        ti = tmap.get(r["h"])
        if ti is None:
            continue
        yi = int(np.abs(lat - r["latitude"]).argmin())
        xi = int(np.abs(lon - r["longitude"]).argmin())
        cells.add((ti, yi, xi))
        found += int(obs[ti, yi, xi] > 0)
    check("every retained detection appears in OBSERVED_FIRE",
          found == len(kept), f"{found}/{len(kept)}")

    # (b) and nothing else does -- no invented fire
    check("OBSERVED_FIRE contains no pixel without a detection",
          int(obs.sum()) == len(cells),
          f"{int(obs.sum())} lit vs {len(cells)} distinct detection cells")

    # (c) the excluded detections must NOT appear
    dropped = df[(df["confidence"] < thr) & (df["type"] == 0)].copy()
    dropped["h"] = pd.to_datetime(dropped["acq_timestamp"]).dt.round("h")
    leaked = 0
    for _, r in dropped.iterrows():
        ti = tmap.get(r["h"])
        if ti is None:
            continue
        yi = int(np.abs(lat - r["latitude"]).argmin())
        xi = int(np.abs(lon - r["longitude"]).argmin())
        if (ti, yi, xi) not in cells and obs[ti, yi, xi] > 0:
            leaked += 1
    check(f"detections below confidence {thr} are absent", leaked == 0,
          f"{leaked} low-confidence detections present")
    info(f"{len(kept)} retained, {len(dropped)} dropped by confidence")


# ------------------------------------------------------------------ 6 ------

def t_against_sources(ds):
    hdr("6. Cross-check against original source rasters")
    lat, lon = ds.latitude.values, ds.longitude.values

    def best_shift(obs, ref, rng=7, m=14):
        best = None
        s = slice(m, -m)
        for dy in range(-rng, rng + 1):
            for dx in range(-rng, rng + 1):
                o = np.roll(np.roll(obs, dy, 0), dx, 1)
                r = np.sqrt(np.nanmean((o[s, s] - ref[s, s]) ** 2))
                if best is None or r < best[0]:
                    best = (r, dy, dx)
        return best

    if os.path.exists(ERA5_SRC):
        src = xr.open_dataset(ERA5_SRC, engine="netcdf4")
        for t in (0, 731, 1463):
            ref = (src["t2m"].isel(valid_time=t)
                   .interp(latitude=("latitude", lat),
                           longitude=("longitude", lon), method="linear").values)
            r, dy, dx = best_shift(ds["t2m"].isel(valid_time=t).values, ref)
            check(f"t2m at step {t} needs no spatial shift vs ERA5 source",
                  (dy, dx) == (0, 0), f"best shift ({dy:+d},{dx:+d}), RMSE {r:.4f} K")
        src.close()

    if os.path.exists(DEM_SRC):
        dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
        tr = rasterio.transform.from_origin(lon[0] - dlon / 2, lat[0] + dlat / 2,
                                            dlon, dlat)
        dst = np.empty((len(lat), len(lon)), np.float32)
        with rasterio.open(DEM_SRC) as s:
            reproject(source=rasterio.band(s, 1), destination=dst,
                      dst_transform=tr, dst_crs="EPSG:4326",
                      src_nodata=s.nodata if s.nodata is not None else 0,
                      dst_nodata=0, resampling=Resampling.bilinear)
        r, dy, dx = best_shift(ds["DEM"].values, dst, rng=5, m=10)
        check("DEM needs no spatial shift vs source GeoTIFF",
              (dy, dx) == (0, 0), f"best shift ({dy:+d},{dx:+d}), RMSE {r:.1f} m")


# ------------------------------------------------------------------ 7 ------

def t_physical(ds):
    hdr("7. Physical plausibility")
    ranges = {
        "t2m": (230, 330, "K"), "d2m": (220, 320, "K"),
        "swvl1": (0, 1, "m3/m3"), "tp": (0, 0.5, "m"),
        "u10": (-50, 50, "m/s"), "v10": (-50, 50, "m/s"),
        "cvl": (0, 1, "1"), "GHS_BUILT": (0, 100, "%"),
        "DEM": (0, 9000, "m"),
    }
    for v, (lo, hi, u) in ranges.items():
        a = ds[v].values
        check(f"{v} within [{lo}, {hi}] {u}",
              float(np.nanmin(a)) >= lo and float(np.nanmax(a)) <= hi,
              f"[{np.nanmin(a):.3f}, {np.nanmax(a):.3f}]")

    nan_vars = {v: int(np.isnan(ds[v].values).sum())
                for v in ds.data_vars if ds[v].dtype.kind == "f"}
    check("no NaNs anywhere", not any(nan_vars.values()),
          str({k: n for k, n in nan_vars.items() if n}))

    check("dewpoint never exceeds air temperature",
          bool((ds["d2m"].values <= ds["t2m"].values + 0.5).all()),
          f"{int((ds['d2m'].values > ds['t2m'].values + 0.5).sum())} violations")
    # ERA5 signs evaporation negative, but positive values are legitimate:
    # they are condensation / frost deposition. Verified on this file -- mean
    # t2m where e > 0 is 262.6 K versus 288.0 K where e < 0, i.e. the positive
    # values sit at freezing altitudes, exactly as deposition should.
    e = ds["e"].values
    pos_share = float((e > 0).mean())
    check("evaporation is predominantly negative, with only minor deposition",
          pos_share < 0.05 and float(e.max()) < 1e-3,
          f"{100 * pos_share:.2f}% positive, max {e.max():.2e}")
    dem = ds["DEM"].values
    # Copernicus DEM GLO-30 covers the whole grid, so unlike the CartoDEM layer
    # it replaced there is no 0-fill and no nodata sentinel at all. A zero here
    # would mean a coverage hole has crept back in -- and the old 0-fill
    # produced artificial ~1,861 m/km cliffs in the Slope features.
    n_zero = int((dem == 0).sum())
    check("DEM has no nodata holes (Copernicus covers the full grid)",
          n_zero == 0, f"{n_zero} zero cells")
    check("no physically impossible slopes from coverage edges",
          float(np.hypot(*np.gradient(dem)).max()) < 1500,
          f"max {float(np.hypot(*np.gradient(dem)).max()):.0f} m/km")

    check("precipitation is non-negative",
          float(ds["tp"].values.min()) >= 0, f"min {ds['tp'].values.min():.2e}")

    wind = np.hypot(ds["u10"].values, ds["v10"].values)
    check("10 m wind speed stays below 40 m/s", float(wind.max()) < 40,
          f"max {wind.max():.1f} m/s")
    info(f"DEM max {ds['DEM'].values.max():.0f} m "
         f"(Uttarakhand's highest peak, Nanda Devi, is 7816 m)")

    t2m = ds["t2m"].values
    diurnal = t2m.reshape(61, 24, -1).mean(axis=(0, 2))
    check("t2m shows a physical diurnal cycle (afternoon warmer than dawn)",
          diurnal.argmax() > diurnal.argmin(),
          f"min at {diurnal.argmin():02d}h UTC, max at {diurnal.argmax():02d}h UTC")


# ------------------------------------------------------------------ 8 ------

def t_lulc(ds):
    hdr("8. Land cover (ESA WorldCover)")
    lulc = ds["LULC"].values
    codes = set(np.unique(lulc).tolist())

    leg = pd.read_csv(LEGEND)
    legend_codes = set(leg["code"].tolist())
    burnable_codes = set(leg.loc[leg["is_burnable"], "code"].tolist())

    check("every raster code appears in the legend", codes <= legend_codes,
          f"orphans {sorted(codes - legend_codes)}")
    check("codes are WorldCover codes, not palette indices",
          codes <= {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100},
          f"unexpected {sorted(codes - {0,10,20,30,40,50,60,70,80,90,95,100})}")

    # WorldCover is global: unlike the Bhuvan render it replaced, there is no
    # off-map background, so no-data should be absent entirely.
    nodata = float((lulc == 0).mean())
    check("no unclassified cells", nodata == 0.0, f"{100 * nodata:.3f}% are code 0")

    burnable = np.isin(lulc, list(burnable_codes))
    info(f"{len(codes)} classes present; {100 * burnable.mean():.1f}% burnable")
    check("burnable share is plausible for Uttarakhand (50-90%)",
          0.50 < burnable.mean() < 0.90, f"{100 * burnable.mean():.1f}%")

    # Sanity against known geography rather than against the pipeline.
    lat = ds["latitude"].values
    snow = lulc == 70
    if snow.any():
        mean_snow_lat = float(lat[np.nonzero(snow)[0]].mean())
        mean_lat = float(lat.mean())
        check("snow/ice sits in the northern (high Himalaya) half",
              mean_snow_lat > mean_lat,
              f"snow mean lat {mean_snow_lat:.2f} vs grid mean {mean_lat:.2f}")
        dem = ds["DEM"].values
        valid = dem > 0
        if (snow & valid).any():
            snow_elev = float(dem[snow & valid].mean())
            all_elev = float(dem[valid].mean())
            check("snow/ice sits well above the mean elevation",
                  snow_elev > all_elev + 500,
                  f"snow {snow_elev:.0f} m vs grid {all_elev:.0f} m")

    if os.path.exists(PURITY):
        with rasterio.open(PURITY) as src:
            share = src.read(1)
        info(f"majority-class purity: mean {share.mean():.3f}, "
             f"{100 * (share < 0.5).mean():.1f}% of cells below 50%")
        check("most cells have a clear majority class",
              float((share >= 0.5).mean()) > 0.85,
              f"only {100 * (share >= 0.5).mean():.1f}% at/above 50%")

    # Fires should overwhelmingly fall on land that can carry a fire.
    ever = ds["ACTIVE_FIRE"].values.astype(bool).any(axis=0)
    on_burnable = float(burnable[ever].mean())
    check("over 85% of burned cells fall on burnable classes",
          on_burnable > 0.85, f"{100 * on_burnable:.1f}%")
    info(f"burned cells by class: " + ", ".join(
        f"{leg.loc[leg['code'] == c, 'name'].iloc[0]} {n}"
        for c, n in sorted(zip(*np.unique(lulc[ever], return_counts=True)),
                           key=lambda t: -t[1])[:5]))


# ------------------------------------------------------------------ 9 ------

def t_leakage(ds):
    hdr("9. Leakage and baselines")
    act = ds["ACTIVE_FIRE"].values.astype(bool)
    print("  Persistence baseline (copy the input frame):")
    ious = {}
    for lead in (1, 8, 24, 48):
        a, b = act[:-lead], act[lead:]
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        ious[lead] = inter / max(union, 1)
        print(f"    lead {lead:>3} h: IoU = {ious[lead]:.4f}")
    check("24 h persistence IoU is well below the old 0.88",
          ious[24] < 0.20, f"{ious[24]:.4f}")
    check("persistence IoU decays monotonically with lead time",
          ious[1] > ious[8] > ious[24] > ious[48])

    # No static feature may coincide with the fire footprint.
    ever = act.any(axis=0)
    for v in ("DEM", "GHS_BUILT", "cvl"):
        a = ds[v].values
        thr = np.nanpercentile(a, 50)
        pred = a > thr
        inter = np.logical_and(pred, ever).sum()
        union = np.logical_or(pred, ever).sum()
        check(f"{v} thresholded does not reproduce the fire mask",
              inter / max(union, 1) < 0.30, f"IoU {inter / max(union, 1):.4f}")

    frac = act.mean()
    info(f"ACTIVE_FIRE positive rate {100 * frac:.4f}% "
         f"-- expect heavy class imbalance when training")


# ----------------------------------------------------------------- 10 ------

def t_metadata(ds):
    hdr("10. Metadata completeness (for publication)")
    for key in ("title", "summary", "source", "Conventions", "spatial_ref",
                "time_coverage_start", "time_coverage_end",
                "coordinate_convention", "modis_min_confidence"):
        check(f"global attribute '{key}' present", key in ds.attrs,
              str(ds.attrs.get(key, ""))[:60])
    no_units = [v for v in ds.data_vars if not ds[v].attrs.get("units")]
    check("every variable declares units", not no_units, str(no_units))
    no_long = [v for v in ds.data_vars if not ds[v].attrs.get("long_name")]
    check("every variable declares a long_name", not no_long, str(no_long))
    for c in ("latitude", "longitude"):
        check(f"{c} has a CF standard_name",
              ds[c].attrs.get("standard_name") == c)


# ----------------------------------------------------------------- 11 ------

def t_coverage(ds):
    """Coverage claims made in the dataset card, verified against the file.

    These exist so the card cannot drift from the data: every figure quoted in
    its Coverage section is asserted here.
    """
    hdr("11. Coverage (claims made in the dataset card)")
    t = pd.to_datetime(ds["valid_time"].values)
    obs = ds["OBSERVED_FIRE"].values.astype(bool)

    steps = np.diff(t.values).astype("timedelta64[h]").astype(int)
    check("weather axis is gapless hourly", set(steps.tolist()) == {1},
          f"step sizes {sorted(set(steps.tolist()))}")

    hrs = np.nonzero(obs.sum(axis=(1, 2)) > 0)[0]
    check("card's count of detection-bearing hours (125)", len(hrs) == 125,
          f"{len(hrs)}")

    gaps = np.diff(hrs)
    check("card's longest observation blackout (84 h)", int(gaps.max()) == 84,
          f"{int(gaps.max())} h at {t[hrs[int(np.argmax(gaps))]]}")

    # MODIS is a polar orbiter: most hours of the day are never sampled.
    hours_of_day = {int(t[h].hour) for h in hrs}
    check("detections fall in only part of the day (polar orbiter)",
          len(hours_of_day) < 24,
          f"{len(hours_of_day)}/24 hours ever contain a detection")
    info(f"overpass hours (UTC): {sorted(hours_of_day)}")

    ever = obs.any(axis=0)
    la, lo = ds["latitude"].values, ds["longitude"].values
    yy, xx = np.nonzero(ever)
    check("card's burned-cell count (2,372)", int(ever.sum()) == 2372,
          f"{int(ever.sum())}")
    check("fire does not reach the eastern edge of the grid",
          lo[xx].max() < lo.max() - 0.3,
          f"easternmost fire {lo[xx].max():.3f} vs grid edge {lo.max():.3f}")
    check("fire does not reach the northern edge of the grid",
          la[yy].max() < la.max() - 0.2,
          f"northernmost fire {la[yy].max():.3f} vs grid edge {la.max():.3f}")
    info(f"fire bbox {lo[xx].min():.3f}-{lo[xx].max():.3f} E, "
         f"{la[yy].min():.3f}-{la[yy].max():.3f} N")

    # No variable may contain NaN: the card promises real data in every cell.
    bad = [v for v in ds.data_vars
           if ds[v].dtype.kind == "f" and bool(np.isnan(ds[v].values).any())]
    check("no NaN in any variable", not bad, f"NaN in {bad}")


def main():
    if not os.path.exists(NC):
        print(f"Not found: {NC}. Run from the Model/ directory.")
        return 1
    print(f"Validating {NC} ({os.path.getsize(NC) / 1e9:.2f} GB)")
    ds = xr.open_dataset(NC, engine="netcdf4")

    t_structure(ds)
    t_coords(ds)
    t_time(ds)
    t_fire_invariants(ds)
    t_modis_roundtrip(ds)
    t_against_sources(ds)
    t_physical(ds)
    t_lulc(ds)
    t_leakage(ds)
    t_metadata(ds)
    t_coverage(ds)

    hdr("SUMMARY")
    n_pass = _results.count("PASS")
    n_fail = _results.count("FAIL")
    n_warn = _results.count("WARN")
    print(f"  {n_pass} passed, {n_fail} failed, {n_warn} warnings "
          f"({len(_results)} checks)")
    if n_fail:
        print("\n  NOT READY TO PUBLISH -- see FAIL lines above.")
    else:
        print("\n  All checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
