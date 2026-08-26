"""
SUPERSEDED. The pipeline now uses ESA WorldCover via worldcover_resample.py.

This script remains only for reproducibility of the earlier draft. Its output
is NOT redistributable: NRSC grants Bhuvan data under a single-user, internal-
use licence with digital databases restricted to authorised government users,
so a derived product may not be published. Nothing it produces is included in
the released dataset. Run it only against your own Bhuvan download, for your
own internal use.

Recover land-cover class codes from the RGB LULC render.

dataset/LULC/UK_LULC50K_2016.tif is not a classified raster: it is a 3-band,
lossily-compressed *picture* of a land-cover map (158k+ distinct RGB triplets in
a 1200x1200 window, where a real thematic product has a few dozen classes).
The previous pipeline took `src.read(1)` -- the red channel -- and treated those
0-255 intensities as class codes, which collapses distinct legend colours that
happen to share a red value and turns JPEG ringing into spurious classes.

This script instead:
  1. finds the legend palette (dominant, well-separated colours),
  2. assigns every full-resolution pixel to its nearest legend colour,
  3. marks pixels too far from any legend colour as UNCLASSIFIED,
  4. downsamples to the 1 km grid by MAJORITY vote (not nearest-neighbour, which
     would pick one arbitrary pixel out of the ~400 covered by each output cell),
  5. writes a single-band uint8 raster plus a legend CSV.

The result is a reconstruction, not the authoritative product. Boundary pixels
are genuinely ambiguous because the source is lossy; the legend CSV records the
per-class pixel share and the run reports the unclassified fraction so the
uncertainty is visible rather than hidden.

Run from the Model/ directory:
    ../.venv/bin/python preprocessing/lulc_classify.py
"""

import csv
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

SRC_PATH = "dataset/LULC/UK_LULC50K_2016.tif"
OUT_TIF = "dataset/resampled-fix/lulc_classified.tif"
OUT_LEGEND = "dataset/LULC/lulc_legend.csv"

# Target grid (same declaration as the other preprocessing scripts)
TARGET_CRS = "EPSG:4326"
TARGET_BOUNDS = (77.5, 28.7, 81.1, 31.5)  # left, bottom, right, top
TARGET_W, TARGET_H = 400, 311

# Palette discovery
# Tuned on this raster: at separation 30 the discovered set is stable at 14
# colours (25 and below splits one legend yellow into two near-duplicates that
# differ only by JPEG ringing; 0.0005 share drops four genuine rare classes).
PALETTE_SIZE = 32          # max legend colours to keep
MIN_SEPARATION = 30        # Chebyshev distance below which two colours are "the same"
MIN_SHARE = 0.0001         # ignore colours covering < 0.01% of the map
AMBIGUITY_TOLERANCE = 40   # Chebyshev distance beyond which a pixel is UNCLASSIFIED

UNCLASSIFIED = 0           # reserved code; also what merge_dynamic's fillna(0) yields

# Colours that are map furniture rather than land cover.
BACKGROUND_COLOURS = [(255, 255, 255), (225, 225, 225)]

ROW_BLOCK = 512            # rows per classification pass, to bound peak memory


def discover_palette(path):
    """Find legend colours from a decimated view of the raster."""
    print(f"Reading decimated view of {path} to find the legend...")
    with rasterio.open(path) as src:
        view = src.read(out_shape=(3, 2048, 2048))
    rgb = view.reshape(3, -1).T.astype(np.int16)
    colours, counts = np.unique(rgb, axis=0, return_counts=True)
    total = counts.sum()

    order = np.argsort(-counts)
    palette, shares = [], []
    for i in order:
        colour, share = colours[i], counts[i] / total
        if share < MIN_SHARE:
            break
        if any(np.abs(colour - p).max() < MIN_SEPARATION for p in palette):
            continue
        palette.append(colour)
        shares.append(share)
        if len(palette) >= PALETTE_SIZE:
            break

    palette = np.array(palette, dtype=np.int16)
    print(f"  found {len(palette)} legend colours "
          f"(covering {100 * sum(shares):.1f}% of sampled pixels exactly)")
    return palette


def classify_full_res(path, palette):
    """Assign every pixel to its nearest legend colour, in row blocks."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        profile = src.profile
        classified = np.zeros((h, w), dtype=np.uint8)
        far_pixels = 0

        for y0 in range(0, h, ROW_BLOCK):
            rows = min(ROW_BLOCK, h - y0)
            block = src.read(window=rasterio.windows.Window(0, y0, w, rows))
            flat = block.reshape(3, -1).T.astype(np.int16)

            # Chebyshev distance to each legend colour; nearest wins.
            dist = np.abs(flat[:, None, :] - palette[None, :, :]).max(axis=2)
            nearest = dist.argmin(axis=1)
            best = dist[np.arange(len(flat)), nearest]

            # Codes are 1-based; 0 stays reserved for UNCLASSIFIED.
            codes = (nearest + 1).astype(np.uint8)
            codes[best > AMBIGUITY_TOLERANCE] = UNCLASSIFIED
            far_pixels += int((best > AMBIGUITY_TOLERANCE).sum())

            classified[y0:y0 + rows] = codes.reshape(rows, w)
            print(f"  classified rows {y0:>5}-{y0 + rows:>5} / {h}", end="\r")

    print()
    print(f"  pixels beyond the ambiguity tolerance: {far_pixels:,} "
          f"({100 * far_pixels / (h * w):.2f}%) -> UNCLASSIFIED")
    return classified, profile


def majority_downsample(classified, profile):
    """Reproject to the 1 km target grid taking the modal class per output cell."""
    print("Downsampling to the target grid by majority vote...")
    dst = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
    reproject(
        source=classified,
        destination=dst,
        src_transform=profile["transform"],
        src_crs=profile["crs"],
        src_nodata=UNCLASSIFIED,
        dst_transform=rasterio.transform.from_bounds(
            *TARGET_BOUNDS, TARGET_W, TARGET_H
        ),
        dst_crs=TARGET_CRS,
        dst_nodata=UNCLASSIFIED,
        resampling=Resampling.mode,
    )
    return dst


def write_outputs(dst, palette):
    os.makedirs(os.path.dirname(OUT_TIF), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_LEGEND), exist_ok=True)

    out_profile = {
        "driver": "GTiff",
        "height": TARGET_H,
        "width": TARGET_W,
        "count": 1,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": rasterio.transform.from_bounds(
            *TARGET_BOUNDS, TARGET_W, TARGET_H
        ),
        "nodata": UNCLASSIFIED,
        "compress": "deflate",
    }
    with rasterio.open(OUT_TIF, "w", **out_profile) as f:
        f.write(dst, 1)
    print(f"Wrote {OUT_TIF}")

    codes, counts = np.unique(dst, return_counts=True)
    share = {int(c): int(n) for c, n in zip(codes, counts)}
    total = dst.size

    with open(OUT_LEGEND, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "R", "G", "B", "is_background",
                         "pixels_1km", "share_1km"])
        writer.writerow([UNCLASSIFIED, "", "", "", "true",
                         share.get(UNCLASSIFIED, 0),
                         f"{share.get(UNCLASSIFIED, 0) / total:.6f}"])
        for i, colour in enumerate(palette):
            code = i + 1
            rgb = tuple(int(v) for v in colour)
            writer.writerow([
                code, rgb[0], rgb[1], rgb[2],
                "true" if rgb in BACKGROUND_COLOURS else "false",
                share.get(code, 0), f"{share.get(code, 0) / total:.6f}",
            ])
    print(f"Wrote {OUT_LEGEND}")

    print("\nClass distribution on the 1 km grid:")
    for code, n in sorted(share.items(), key=lambda kv: -kv[1])[:15]:
        if code == UNCLASSIFIED:
            label = "UNCLASSIFIED / background"
        else:
            rgb = tuple(int(v) for v in palette[code - 1])
            label = f"RGB{rgb}"
            if rgb in BACKGROUND_COLOURS:
                label += "  (map background, not land cover)"
        print(f"  code {code:>3}: {n:>7,} px ({100 * n / total:5.2f}%)  {label}")


def main():
    if not os.path.exists(SRC_PATH):
        print(f"Not found: {SRC_PATH}\nRun this from the Model/ directory.")
        return 1
    palette = discover_palette(SRC_PATH)
    classified, profile = classify_full_res(SRC_PATH, palette)
    dst = majority_downsample(classified, profile)
    write_outputs(dst, palette)
    print("\nDone. Point merge_dynamic.LULC_PATH at the classified raster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
