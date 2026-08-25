"""
Pre-publication audit of final_feature_stack_DYNAMIC_interpolated.nc

Checks the properties that matter when the file leaves this machine and other
people train on it: georeferencing, label provenance, temporal leakage, and
whether the stored values still mean what the source data meant.

Run from the Model/ directory:
    ../.venv/bin/python dataset-validation-scripts/audit_release_readiness.py
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

NC_PATH = "dataset/final_feature_stack_DYNAMIC_interpolated.nc"
PREV_NC_PATH = "dataset/final_feature_stack_DYNAMIC_new.nc"
MODIS_CSV = "dataset/MODIS/final-modis.csv"
LULC_TIF = "dataset/resampled-fix/lulc_resampled.tif"

# Grid the preprocessing scripts declared as the target
DECLARED_BOUNDS = (77.5, 28.7, 81.1, 31.5)  # left, bottom, right, top
DECLARED_RES = 0.009
DECLARED_W, DECLARED_H = 400, 311


def hdr(n, title):
    print(f"\n{'=' * 72}\n[{n}] {title}\n{'=' * 72}")


def check_grid(ds):
    hdr(1, "Georeferencing: are the coordinate labels cell centers?")
    lat = ds.latitude.values
    lon = ds.longitude.values
    left, bottom, right, top = DECLARED_BOUNDS

    dlat = np.abs(np.diff(lat))
    dlon = np.abs(np.diff(lon))
    print(f"  stored lat spacing: {dlat.mean():.9f} (uniform: {np.ptp(dlat) < 1e-12})")
    print(f"  stored lon spacing: {dlon.mean():.9f} (uniform: {np.ptp(dlon) < 1e-12})")

    # What era5-resample.py *should* have produced: from_bounds() makes cells of
    # size (extent / N), with centers offset half a cell inside the bounds.
    true_dlat = (top - bottom) / DECLARED_H
    true_dlon = (right - left) / DECLARED_W
    print(f"  true cell size from bbox/N: dlat={true_dlat:.9f} dlon={true_dlon:.9f}")

    # What linspace(top, bottom, H) actually produces: extent / (N-1)
    linspace_dlat = (top - bottom) / (DECLARED_H - 1)
    linspace_dlon = (right - left) / (DECLARED_W - 1)
    print(f"  linspace(endpoint) spacing: dlat={linspace_dlat:.9f} dlon={linspace_dlon:.9f}")

    lat_is_linspace = np.isclose(dlat.mean(), linspace_dlat, rtol=1e-9)
    lon_is_linspace = np.isclose(dlon.mean(), linspace_dlon, rtol=1e-9)
    print(f"\n  -> lat matches linspace-endpoint (WRONG, edges not centers): {lat_is_linspace}")
    print(f"  -> lon matches linspace-endpoint (WRONG, edges not centers): {lon_is_linspace}")

    if lat_is_linspace or lon_is_linspace:
        # Offset at the top-left corner, and cumulative drift across the grid
        true_lat0 = top - true_dlat / 2
        true_lon0 = left + true_dlon / 2
        off_lat0 = abs(lat[0] - true_lat0)
        off_lon0 = abs(lon[0] - true_lon0)
        # drift = spacing error accumulated over the full grid
        drift_lat = abs(linspace_dlat - true_dlat) * len(lat)
        drift_lon = abs(linspace_dlon - true_dlon) * len(lon)
        km = 111.0
        print(f"\n  corner offset : {off_lat0:.6f} deg lat ({off_lat0 * km:.3f} km), "
              f"{off_lon0:.6f} deg lon ({off_lon0 * km:.3f} km)")
        print(f"  cumulative drift across grid: {drift_lat:.6f} deg lat "
              f"({drift_lat * km:.3f} km), {drift_lon:.6f} deg lon ({drift_lon * km:.3f} km)")
        print(f"  (cell is ~{true_dlat * km:.3f} km, so drift is "
              f"~{drift_lat / true_dlat:.2f} cells in lat, {drift_lon / true_dlon:.2f} in lon)")

    print(f"\n  grid shape stored: {len(lat)} x {len(lon)}")
    print(f"  grid shape declared by resample scripts: {DECLARED_H} x {DECLARED_W}")
    if len(lat) == DECLARED_H - 3 or len(lon) == DECLARED_W - 3:
        print("  note: shrinkage is the DEM crop; see check 6 for the off-by-one.")


def check_lulc(ds):
    hdr(2, "LULC: is the categorical layer still categorical?")
    lulc = ds["LULC"].values
    print(f"  stored dtype: {ds['LULC'].dtype}")
    u = np.unique(lulc)
    print(f"  unique values: {len(u)}")
    print(f"  first 20: {u[:20]}")
    frac = u[(u != np.floor(u))]
    print(f"\n  -> non-integer class codes present: {len(frac)} of {len(u)}")
    if len(frac):
        print(f"     examples: {frac[:10]}")
        print("     Non-integer LULC codes are not valid classes: nearest-neighbour")
        print("     resampling should have preserved integers exactly.")
    else:
        print("     All values are integral (categories survived resampling).")


def check_fire_provenance(ds):
    hdr(3, "Fire label: which script produced the published mask?")
    fire = ds["MODIS_FIRE_T1"].values
    T = fire.shape[0]
    per_frame = fire.sum(axis=(1, 2))
    print(f"  total fire pixel-hours: {fire.sum():,.0f}")
    print(f"  frames with any fire: {(per_frame > 0).sum()} / {T}")
    print(f"  unique values: {np.unique(fire)}")

    # Signature test: fix_dataset_persistence.py forward-fills each ignited pixel
    # for exactly 24 steps. interpolate_fire_morphology.py grows a blob via EDT
    # and never lets it shrink.
    lit = fire > 0
    # run lengths of the first 200 pixels that ever ignite
    ys, xs = np.where(lit.any(axis=0))
    runs = []
    for y, x in list(zip(ys, xs))[:400]:
        col = lit[:, y, x]
        d = np.diff(col.astype(np.int8))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if col[0]:
            starts = np.r_[0, starts]
        if col[-1]:
            ends = np.r_[ends, T]
        for s, e in zip(starts, ends):
            runs.append(e - s)
    runs = np.array(runs)
    if len(runs):
        vals, cnts = np.unique(runs, return_counts=True)
        top = sorted(zip(cnts, vals), reverse=True)[:5]
        print(f"\n  burn-run lengths (sampled pixels): n={len(runs)}")
        print(f"  most common run lengths (count, hours): {top}")
        exact24 = (runs == 24).mean()
        print(f"  -> fraction of runs exactly 24h: {exact24:.3f}")
        if exact24 > 0.5:
            print("     Signature matches fix_dataset_persistence.py (24h forward-fill),")
            print("     NOT interpolate_fire_morphology.py (EDT dilation).")

    hdr(4, "Fire label: does anything ever stop burning?")
    print(f"  per-frame fire count, first 48h: {per_frame[:48].astype(int)}")
    drops = np.diff(per_frame)
    print(f"  frames where total fire decreases: {(drops < 0).sum()} / {T - 1}")
    print(f"  frames where total fire increases: {(drops > 0).sum()} / {T - 1}")
    print(f"  max fire in any frame: {per_frame.max():.0f} px "
          f"({100 * per_frame.max() / (fire.shape[1] * fire.shape[2]):.3f}% of map)")
    if (drops < 0).sum() == 0:
        print("  -> Fire is MONOTONIC: no pixel ever extinguishes. The mask is a")
        print("     cumulative burned-area layer, not an active-fire layer.")


def check_leakage(ds):
    hdr(5, "Temporal leakage: how much of frame T+k is just frame T?")
    fire = ds["MODIS_FIRE_T1"].values > 0.5
    for k in (1, 8, 24, 48):
        a = fire[:-k]
        b = fire[k:]
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        iou = inter / max(union, 1)
        # recall of persistence: of the fire at T+k, how much was already lit at T
        recall = inter / max(b.sum(), 1)
        print(f"  lead {k:>3}h: persistence IoU = {iou:.4f}   "
              f"(fraction of target already lit at T = {recall:.4f})")
    print("\n  A model that copies its input scores the IoU above. Any reported")
    print("  score at or below it is worthless; this is the real baseline.")


def check_crop_offbyone(ds):
    hdr(6, "Crop off-by-one in merge_dynamic.py")
    # merge_dynamic crops with slice(min_idx, max_idx) which drops the last valid
    # row/column. Detect by checking whether the DEM still has valid data hard
    # against the final edge.
    dem = ds["DEM"].values
    print(f"  stored shape: {dem.shape}")
    print(f"  DEM>0 on first row: {(dem[0] > 0).mean():.3f}, last row: {(dem[-1] > 0).mean():.3f}")
    print(f"  DEM>0 on first col: {(dem[:, 0] > 0).mean():.3f}, last col: {(dem[:, -1] > 0).mean():.3f}")
    print("\n  merge_dynamic.py:191 uses slice(lat_min_idx, lat_max_idx).")
    print("  Python slices are half-open, so the last valid row and column found")
    print("  by np.argwhere are excluded from the published file.")


def check_edge_clamping(ds):
    hdr(7, "Out-of-bounds fire clamped onto the grid edge")
    fire = ds["MODIS_FIRE_T1"].values > 0.5
    ever = fire.any(axis=0)
    H, W = ever.shape
    edge = ever[0].sum() + ever[-1].sum() + ever[:, 0].sum() + ever[:, -1].sum()
    edge_cells = 2 * H + 2 * W
    interior = ever.sum() - edge
    interior_cells = H * W - edge_cells
    print(f"  burned cells on the 1-px border: {edge} / {edge_cells} border cells "
          f"({100 * edge / edge_cells:.2f}%)")
    print(f"  burned cells in the interior:    {interior} / {interior_cells} "
          f"({100 * interior / interior_cells:.2f}%)")
    if edge_cells and interior_cells:
        ratio = (edge / edge_cells) / max(interior / interior_cells, 1e-9)
        print(f"  -> border is {ratio:.1f}x as likely to have burned as the interior")
        if ratio > 2:
            print("     merge_dynamic.py:131 np.clip()s out-of-domain MODIS points onto")
            print("     the edge instead of discarding them, which manufactures fire there.")


def check_modis_mapping(ds):
    hdr(8, "MODIS point -> pixel mapping (searchsorted vs nearest)")
    if not os.path.exists(MODIS_CSV):
        print("  MODIS csv not found; skipping.")
        return
    df = pd.read_csv(MODIS_CSV)
    lats = ds.latitude.values
    lons = ds.longitude.values
    p_lat = df["latitude"].values
    p_lon = df["longitude"].values

    # Reproduce merge_dynamic.py's index computation exactly
    if lats[1] > lats[0]:
        ss_lat = np.searchsorted(lats, p_lat)
    else:
        ss_lat = len(lats) - 1 - np.searchsorted(lats[::-1], p_lat)
    if lons[1] > lons[0]:
        ss_lon = np.searchsorted(lons, p_lon)
    else:
        ss_lon = len(lons) - 1 - np.searchsorted(lons[::-1], p_lon)
    ss_lat_c = np.clip(ss_lat, 0, len(lats) - 1)
    ss_lon_c = np.clip(ss_lon, 0, len(lons) - 1)

    # True nearest index
    nn_lat = np.abs(lats[None, :] - p_lat[:, None]).argmin(axis=1)
    nn_lon = np.abs(lons[None, :] - p_lon[:, None]).argmin(axis=1)

    mism_lat = (ss_lat_c != nn_lat)
    mism_lon = (ss_lon_c != nn_lon)
    print(f"  MODIS detections: {len(df)}")
    print(f"  lat index differs from true nearest: {mism_lat.sum()} "
          f"({100 * mism_lat.mean():.1f}%)")
    print(f"  lon index differs from true nearest: {mism_lon.sum()} "
          f"({100 * mism_lon.mean():.1f}%)")
    both = (mism_lat | mism_lon)
    print(f"  detections landing in the wrong cell: {both.sum()} "
          f"({100 * both.mean():.1f}%)")

    # Signed bias: searchsorted rounds toward one side systematically
    bias_lat = (ss_lat_c.astype(int) - nn_lat.astype(int))
    bias_lon = (ss_lon_c.astype(int) - nn_lon.astype(int))
    print(f"  signed lat index bias: mean={bias_lat.mean():+.3f} cells")
    print(f"  signed lon index bias: mean={bias_lon.mean():+.3f} cells")
    print("  (a nonzero mean is a systematic geolocation shift, not random noise)")

    # How many points fall outside the domain and get clamped
    out = ((p_lat > lats.max()) | (p_lat < lats.min()) |
           (p_lon > lons.max()) | (p_lon < lons.min()))
    print(f"\n  detections outside the cropped grid: {out.sum()} "
          f"({100 * out.mean():.1f}%) -> clamped onto the border, not dropped")


def check_modis_filtering():
    hdr(9, "MODIS source filtering (confidence / type)")
    if not os.path.exists(MODIS_CSV):
        print("  MODIS csv not found; skipping.")
        return
    df = pd.read_csv(MODIS_CSV)
    print(f"  rows: {len(df)}")
    if "type" in df:
        vt = df["type"].value_counts().sort_index()
        names = {0: "presumed vegetation fire", 1: "active volcano",
                 2: "other static land source", 3: "offshore"}
        print("  type breakdown:")
        for k, v in vt.items():
            print(f"    {k} ({names.get(k, '?')}): {v}")
        non_veg = (df["type"] != 0).sum()
        print(f"  -> non-vegetation detections kept as fire labels: {non_veg}")
    if "confidence" in df:
        c = df["confidence"]
        print(f"  confidence: min={c.min()} median={c.median()} max={c.max()}")
        for thr in (30, 50, 80):
            print(f"    below {thr}: {(c < thr).sum()} ({100 * (c < thr).mean():.1f}%)")
        print("  -> clean-datasets/modis.py applies no confidence threshold.")
    if "daynight" in df:
        print(f"  daynight: {df['daynight'].value_counts().to_dict()}")


def check_static_stored_as_dynamic(ds):
    hdr(10, "Storage waste: static fields stored with a time axis")
    for var in ds.data_vars:
        da = ds[var]
        if "valid_time" not in da.dims:
            continue
        a = da.isel(valid_time=0).values
        b = da.isel(valid_time=len(ds.valid_time) // 2).values
        c = da.isel(valid_time=-1).values
        constant = np.array_equal(np.nan_to_num(a), np.nan_to_num(b)) and \
                   np.array_equal(np.nan_to_num(a), np.nan_to_num(c))
        if constant:
            nbytes = da.nbytes / 1e6
            print(f"  {var:>14}: CONSTANT over time but stored {nbytes:,.0f} MB "
                  f"({len(ds.valid_time)} identical copies)")


def check_encoding(ds):
    hdr(11, "Compression / encoding of the published file")
    for var in ds.data_vars:
        enc = ds[var].encoding
        print(f"  {var:>14}: dtype={enc.get('dtype')} "
              f"zlib={enc.get('zlib')} complevel={enc.get('complevel')} "
              f"chunks={enc.get('chunksizes')}")
    size = os.path.getsize(NC_PATH) / 1e9
    raw = sum(ds[v].nbytes for v in ds.data_vars) / 1e9
    print(f"\n  on-disk: {size:.2f} GB   uncompressed: {raw:.2f} GB   "
          f"ratio: {raw / size:.2f}x")
    if raw / size < 1.2:
        print("  -> Effectively uncompressed. interpolate_fire_morphology.py and")
        print("     fix_dataset_persistence.py both call to_netcdf() with no encoding=,")
        print("     dropping the zlib settings merge_dynamic.py had applied.")


def check_nans_and_ranges(ds):
    hdr(12, "NaNs and physical value ranges")
    for var in ds.data_vars:
        da = ds[var]
        v = da.values
        n_nan = int(np.isnan(v).sum())
        print(f"  {var:>14}: min={np.nanmin(v):12.4f} max={np.nanmax(v):12.4f} "
              f"mean={np.nanmean(v):12.4f} nans={n_nan}")
    if "t2m" in ds:
        t = ds["t2m"].values
        print(f"\n  t2m looks like {'Kelvin' if np.nanmean(t) > 200 else 'Celsius'}")
    if "tp" in ds:
        tp = ds["tp"].values
        print(f"  tp (total precipitation) negative values: {int((tp < 0).sum())}")
    if "GHS_BUILT" in ds:
        g = ds["GHS_BUILT"].values
        print(f"  GHS_BUILT >100 (should be 0-100% built surface): {int((g > 100).sum())}")


def check_time_axis(ds):
    hdr(13, "Time axis integrity")
    t = ds.valid_time.values
    d = np.diff(t) / np.timedelta64(1, "h")
    print(f"  steps: {len(t)}   {t[0]} .. {t[-1]}")
    print(f"  unique step sizes (hours): {np.unique(d)}")
    print(f"  duplicated timestamps: {len(t) - len(np.unique(t))}")
    print(f"  monotonic increasing: {bool(np.all(d > 0))}")


def main():
    if not os.path.exists(NC_PATH):
        print(f"Not found: {NC_PATH}\nRun this from the Model/ directory.")
        return
    print(f"Auditing {NC_PATH} ({os.path.getsize(NC_PATH) / 1e9:.2f} GB)")
    ds = xr.open_dataset(NC_PATH, engine="h5netcdf")

    check_grid(ds)
    check_lulc(ds)
    check_fire_provenance(ds)
    check_leakage(ds)
    check_crop_offbyone(ds)
    check_edge_clamping(ds)
    check_modis_mapping(ds)
    check_modis_filtering()
    check_static_stored_as_dynamic(ds)
    check_encoding(ds)
    check_nans_and_ranges(ds)
    check_time_axis(ds)
    print(f"\n{'=' * 72}\nAudit complete.\n{'=' * 72}")


if __name__ == "__main__":
    main()
