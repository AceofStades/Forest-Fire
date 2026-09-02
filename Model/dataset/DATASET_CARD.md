# Uttarakhand Wildfire Feature Stack (1 km, hourly, Apr–May 2016)

Co-registered weather, terrain, land cover and MODIS active-fire labels on a
single ~1 km grid over Uttarakhand, India, at hourly resolution.

**File:** `final_feature_stack_RELEASE_v2.nc` — 2.89 GB (NetCDF4, zlib level 1 + shuffle)
**Grid:** 311 × 400 cells, EPSG:4326, cell size 0.009003° lat × 0.009000° lon (~1 km)
**Extent:** 77.5–81.1 °E, 28.7–31.5 °N
**Time:** 1,464 hourly steps, 2016-04-01 00:00 to 2016-05-31 23:00 UTC
**Coordinates are cell centres.**

## Variables

| Name | Dims | Units | Source |
|---|---|---|---|
| `t2m` | time, lat, lon | K | ERA5-Land 2 m air temperature |
| `d2m` | time, lat, lon | K | ERA5-Land 2 m dewpoint |
| `u10` / `v10` | time, lat, lon | m s⁻¹ | ERA5-Land 10 m wind components |
| `swvl1` | time, lat, lon | m³ m⁻³ | ERA5-Land soil water, 0–7 cm |
| `tp` | time, lat, lon | m | ERA5-Land total precipitation |
| `e` | time, lat, lon | m w.e. | ERA5-Land total evaporation |
| `cvl` | lat, lon | 1 | ERA5-Land low vegetation cover (static) |
| `DEM` | lat, lon | m | Copernicus DEM GLO-30 (2021), averaged per cell |
| `LULC` | lat, lon | class code (uint8) | ESA WorldCover 10 m v200 (2021), areal majority per cell |
| `GHS_BUILT` | lat, lon | % | GHSL GHS-BUILT-S R2023A (2018) |
| `OBSERVED_FIRE` | time, lat, lon | 0/1 (int8) | MODIS detections at their observed hour |
| `ACTIVE_FIRE` | time, lat, lon | 0/1 (int8) | Detections + 12 h causal persistence |
| `BURNED_AREA` | time, lat, lon | 0/1 (int8) | Cumulative burn scar |

## The three fire channels

Pick deliberately — they answer different questions.

- **`OBSERVED_FIRE`** is exactly what MODIS saw, at the hour it saw it. No
  persistence, no interpolation, nothing invented. Sparse: 2,597 pixel-hours.
- **`ACTIVE_FIRE`** keeps a detected cell alight for 12 hours, then extinguishes.
  The fill is strictly forward in time, so no frame is ever constructed from an
  observation that has not happened yet. 30,795 pixel-hours (11.9× observed).
  **This is the intended prediction target.**
- **`BURNED_AREA`** is the running maximum of `ACTIVE_FIRE`: has this cell burned
  at any point up to now. Monotone by definition — that is what a burn scar is.
  Do not use it as a next-step prediction target; it can only grow.

### Persistence baseline — read this before reporting a score

A model that simply copies its input frame to its output scores:

| Lead time | Persistence IoU on `ACTIVE_FIRE` |
|---|---|
| 1 h | 0.8499 |
| 8 h | 0.2165 |
| 24 h | 0.0105 |

Any reported IoU at or below the figure for your lead time is not evidence of
skill. At 1 h the task is nearly trivial; 8 h is the most informative setting.
At 24 h the 12 h persistence window has fully elapsed, so the task becomes
predicting *new* ignitions rather than tracking existing fire — expect low
absolute IoU and judge against the 0.0105 baseline, not against 1.0.

## Provenance and processing

- **ERA5-Land** (0.1°) bilinearly resampled to the 1 km grid. Grid points are
  treated as cell centres, and the raster transform is built from the outer
  corner accordingly.
- **MODIS active fire** (FIRMS) filtered to `type == 0` (presumed vegetation
  fire) and `confidence >= 30`; 2,634 of 2,828 detections retained. Points are
  assigned to the *nearest* grid cell. Detections outside the domain are dropped,
  not clamped to the border.
- **DEM** from Copernicus DEM GLO-30 (~30 m), *averaged* to 1 km — about
  1,100 source pixels per cell; a point sample would throw most away.
- **GHS-BUILT** reprojected to the target grid with explicit nodata.
- **Land cover** from ESA WorldCover 10 m v200 (2021). Downsampling to 1 km is
  100x in each direction, so ~11,664 source pixels fall in every target cell.
  Each cell takes the **areal majority** class — not a nearest-neighbour sample,
  which would keep one source pixel in ten thousand and discard the rest.
  `worldcover_legend.csv` gives each code its name and a burnable flag.

## Caveats

**Land cover is a majority class, so mixed cells lose their minority.** At 1 km
each cell is a single WorldCover class covering ~117 km2 of 10 m source. Mean
purity of the winning class is **0.755** (median 0.766), and **8.3% of cells**
have no class above 50% — the majority there is a plurality. If that
matters, regenerate
per-class fractions with `preprocessing/worldcover_resample.py`, which computes
them internally before taking the argmax.

**Every cell carries a real class.** There is no off-map background and no
unclassified fill: 0.0% of the grid is code 0. **99.9% of cells that ever burn
fall on burnable classes** (tree cover 2,125, cropland 194, grassland 50) — a
useful independent check that the layer is correctly georeferenced.

**Single region, single season.** April–May 2016 only. Do not expect these
relationships to transfer to other regions or seasons without validation.

**MODIS detections are not burned area.** Each is a ~1 km footprint flagged as
containing thermal anomaly, not a mapped fire perimeter. Cloud and canopy cover
cause missed detections, and overpass timing (~4/day) bounds temporal resolution.


## Coverage

**Spatial.** 311 x 400 cells at ~1 km. The grid is a *rectangle* drawn around
Uttarakhand, so it also contains slivers of
Himachal Pradesh, Uttar Pradesh, Nepal and Tibet. No state boundary mask is
included — if you need one, bring your own polygon. Every cell carries real
data in every variable; there are no NaNs anywhere in the file.

Fire is far from uniform across that rectangle:

| | Extent |
|---|---|
| Grid | 77.504-81.095 E, 28.705-31.495 N |
| Cells that ever burn | 77.504-80.555 E, 28.705-31.099 N |

The eastern ~13% of the grid and the northern strip **never burn** in this
period — the north is high Himalaya (snow, ice, bare rock). Only **1.91% of
cells** (2,372 of 124,400) ever burn. A spatial split that puts the east or
the north in validation will hand you a fold with almost no positives.

**Temporal.** 1,464 consecutive hourly steps, 2016-04-01 00:00 to
2016-05-31 23:00 UTC. Zero gaps, zero duplicate timestamps, uniform 1 h
spacing. The *weather* really is hourly.

**The label is not.** MODIS is a polar orbiter, so detections exist only when
a satellite passed over:

| | |
|---|---|
| Hours containing a detection | 125 of 1,464 (8.5%) |
| Median gap between them | 9 h |
| Longest blackout | **84 h** (2016-05-23 08:00 -> 2016-05-26 20:00) |
| Hours of day ever containing a detection | 10 of 24 |

Detections cluster in two windows — 05:00-09:00 UTC (10:30-14:30 IST, the
Terra/Aqua daytime passes) and 16:00-21:00 UTC (the night passes). The single
busiest hour, 08:00 UTC, holds **41%** of all detections.

Two consequences worth designing around:

1. **Fourteen hours of the day contain no detection, ever.** A prediction for
   03:00 UTC cannot be scored against a true positive, because none exists at
   that hour anywhere in the record. Either restrict evaluation to overpass
   hours, or accept that most of your timeline is unlabelled rather than
   negative.
2. **Absence of fire is not evidence of no fire.** A zero in `OBSERVED_FIRE`
   between overpasses means "not observed", not "not burning". `ACTIVE_FIRE`'s
   12 h persistence exists precisely to bridge that, which is why it — and not
   `OBSERVED_FIRE` — is the intended target.

During the 84 h blackout in late May, `ACTIVE_FIRE` decays to zero after 12 h
and stays there for three days. That is missing observation, not extinguished
fire.

## Reading it

```python
import xarray as xr
ds = xr.open_dataset("final_feature_stack_RELEASE_v2.nc")
```

Let xarray choose the engine. Kaggle's image ships `h5netcdf` but **not**
`netcdf4`, so pinning `engine="netcdf4"` raises `unrecognized engine` there.
Both read this file identically — same dtypes, same values.

If you also *write* in the same process, stick to one engine: `netCDF4` and
`h5netcdf` bundle separate HDF5 libraries and mixing them breaks dimension-scale
calls. Reading alone is unaffected.

On FUSE mounts (ntfs-3g, sshfs) set `HDF5_USE_FILE_LOCKING=FALSE` first.

## Reproducing

From `Model/`:

```bash
python preprocessing/worldcover_resample.py         # WorldCover 10 m -> 1 km majority
python preprocessing/resampling/era5-resample.py    # ERA5-Land -> 1 km grid
python preprocessing/merge_dynamic.py               # merge everything -> RELEASE.nc
python dataset-validation-scripts/test_preprocessing_fixes.py   # regression tests
python dataset-validation-scripts/validate_release.py           # 86 checks against the sources
```

## Citation

Cite the archived copy. The Zenodo DOI needs no Kaggle account and always
resolves to the current version:

> Bokade, R., Barai, V., Bhogle, S., & Chapherkar, Y. (2026). *Uttarakhand
> Wildfire Dataset (April-May 2016)* [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.22131871

## Licence and attribution

Derived from five sources, each redistributable with attribution:

- **ERA5-Land** — Copernicus Climate Change Service (C3S) / ECMWF.
- **MODIS active fire (FIRMS)** — NASA.
- **Copernicus DEM GLO-30** — © DLR e.V. 2010-2014 and © Airbus Defence and
  Space GmbH 2014-2018, provided under COPERNICUS by the European Union and
  ESA; free for any use with attribution.
- **ESA WorldCover 10 m v200 (2021)** — (c) ESA WorldCover project 2021.
  Contains modified Copernicus Sentinel data (2021), processed by the ESA
  WorldCover consortium. Licensed **CC-BY-4.0**.
- **GHSL GHS-BUILT-S R2023A** — European Commission JRC, CC-BY-4.0.

An earlier draft used Bhuvan/NRSC products: LULC 50K and CartoDEM elevation
tiles. Both were removed:
NRSC grants a single-user, internal-use licence with digital databases
restricted to authorised government users, which does not permit redistributing
a derived product. **No NRSC data is present in this file.**

Replacing CartoDEM also fixed a bug: its 0-filled nodata holes created
artificial 1,861 m/km cliffs (vs 141 m/km elsewhere) in the slope features.
