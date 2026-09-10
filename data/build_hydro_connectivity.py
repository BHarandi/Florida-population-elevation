"""
Author: Bella Harandi
Date: Septermber 2026
One-time ETL: turn the raw elevation DEM into a hydro-connectivity threshold
raster for the Sea Level Rise tab.

Problem: a plain "elevation <= sea_level_m" bathtub model (what get_flood_overlay
and compute_population_at_risk used before this script existed) marks any low
pixel as flooded even if it's an inland depression with no path to the ocean —
overstating flood extent and population at risk. Real flooding requires a
continuous chain of at-or-below-threshold cells connecting the pixel back to
open water.

Method: multi-source region growing from the ocean, sweeping sea level upward
in small steps. At each step, a pixel is newly "connected" once it is both
(a) at or below the current level and (b) 4-connected to a pixel that's
already connected. The level at which that first happens is recorded as the
pixel's hydro-connectivity threshold — the minimum sea level rise needed for
water to actually reach it. Comparing that threshold against a scenario's sea
level (instead of comparing raw elevation) gives a hydrologically-connected
flood mask "for free" at runtime, with no per-request recomputation.

Ocean seed: dem_florida_100m.tif fills open water with a flat 0.0 m rather
than real bathymetry (68% of all pixels), and inland lakes are NOT flattened
this way (e.g. Lake Okeechobee reads 2.28 m) — so the connected components of
"elevation <= 0" double as a free ocean mask. The two largest components
(Gulf of Mexico + Atlantic) cover ~99% of that area; small leftover components
are isolated ponds/canals and are excluded from the seed.

DEM quirk: dem_florida_100m.tif encodes missing data as a literal -999999
value without declaring a `nodata` tag, so those pixels silently read as
"far below sea level" downstream. This script treats anything below
DEM_NODATA_SENTINEL as missing and gives it a real nodata value in the output.

Output: data/hydro_connect_threshold_m.tif — same grid/CRS/shape as the DEM.
Value = minimum sea level (m, NAVD88) at which a pixel is ocean-connected;
NEVER_FLOODS for land above the highest modeled level; nodata for gaps in the
source DEM.

Run: python build_hydro_connectivity.py   (needs scipy; not a dashboard runtime dep)
"""
import os
import time

import numpy as np
import rasterio
from scipy.ndimage import label

_BASE = os.path.dirname(os.path.abspath(__file__))
DEM_PATH = os.path.join(_BASE, "data", "dem_florida_100m.tif")
OUT_PATH = os.path.join(_BASE, "data", "hydro_connect_threshold_m.tif")

DEM_NODATA_SENTINEL = -1000.0   # dem_florida_100m.tif's real gap value is -999999
NEVER_FLOODS         = 9999.0   # land never connects to the ocean within MAX_LEVEL_M
MAX_LEVEL_M           = 60.0    # matches the dashboard's manual SLR slider (0-60 ft / 0-60 m)
SEED_COVERAGE         = 0.99    # keep the largest sea-level components covering this much area

# Finer steps at realistic sea-level-rise magnitudes, coarser further out where
# the slider's range is more "what if" than plausible this century.
STEP_SCHEDULE = [
    (0.0, 5.0, 0.05),
    (5.0, 15.0, 0.25),
    (15.0, MAX_LEVEL_M, 1.0),
]

STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # 4-connectivity (rook)


def level_steps():
    levels = []
    for lo, hi, step in STEP_SCHEDULE:
        n = int(round((hi - lo) / step))
        levels.extend(lo + step * i for i in range(1, n + 1))
    return sorted(set(round(v, 6) for v in levels))


def main():
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1)
        profile = src.profile

    valid = dem > DEM_NODATA_SENTINEL

    sea_mask = valid & (dem <= 0.0)
    labeled, _ = label(sea_mask, structure=STRUCT4)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    cum = np.cumsum(sizes[order]) / sizes.sum()
    n_seed = int(np.searchsorted(cum, SEED_COVERAGE) + 1)
    seed_labels = order[:n_seed]
    print(f"Ocean seed: {n_seed} component(s) covering {cum[n_seed - 1] * 100:.1f}% "
          f"of at-or-below-MSL area")

    connected = np.isin(labeled, seed_labels)
    threshold = np.full(dem.shape, NEVER_FLOODS, dtype=np.float32)
    threshold[connected] = 0.0

    levels = level_steps()
    t0 = time.time()
    for i, level in enumerate(levels):
        candidate = valid & (dem <= level)
        if not (candidate & ~connected).any():
            continue
        labeled, _ = label(candidate, structure=STRUCT4)
        seed_ids = np.unique(labeled[connected & candidate])
        seed_ids = seed_ids[seed_ids != 0]
        newly = np.isin(labeled, seed_ids) & ~connected
        threshold[newly] = level
        connected |= newly
        if i % 20 == 0 or i == len(levels) - 1:
            print(f"  level {level:6.2f} m ({i + 1}/{len(levels)}): "
                  f"{connected.sum():,} px connected, {time.time() - t0:.1f}s elapsed")

    threshold[~valid] = np.nan

    profile.update(dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(OUT_PATH, "w", **profile) as dst:
        dst.write(threshold, 1)
    print(f"Wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
