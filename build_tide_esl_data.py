"""
One-time ETL: filter the nationwide BAYEX extreme-sea-level (ESL) NetCDF down
to Florida sites and reshape into a compact parquet the dashboard can load
without needing netCDF4 as a runtime dependency (mirrors the existing
population_by_elevation.parquet / Florida_Hazards_*.parquet pattern).

Source: data/tide/BAYEX-TG-EXT_RL_ESL_PRED.nc
  - 1,712 sites nationwide, lat 17.7-71.4 N (all US coasts)
  - dims: statistics(3) x return_periods(999, years 2..1000) x sites(1712)
  - 4 variables (methods), units = meters relative to MSL:
      BAYEX_RL_ESL_CONV_TPXO   convolution of TPXO tidal peak + BAYEX skew surge
      BAYEX_RL_ESL_MHW_TPXO    TPXO MHW + BAYEX skew-surge return level
      BAYEX_RL_ESL_MHHW_TPXO   TPXO MHHW + BAYEX skew-surge return level
      BAYEX_RL_ESL_HAT_TPXO    TPXO HAT + BAYEX skew-surge return level
  - "statistics" dim order assumed [central estimate, lower bound, upper bound]
    based on value ordering (e.g. one site's 2-yr CONV values were
    [3.34, 3.29, 3.39] -> index 0 sits between 1 and 2). Re-check against the
    BAYEX source documentation if that assumption ever looks wrong.

Output: data/esl_return_levels_fl.parquet
  one row per (site, return_period), columns:
    site_id, lat, lon, return_period,
    esl_conv_central/lower/upper, esl_mhw_..., esl_mhhw_..., esl_hat_...

Run: python build_tide_esl_data.py
"""
import os
import numpy as np
import pandas as pd
import netCDF4 as nc

_BASE = os.path.dirname(os.path.abspath(__file__))
NC_PATH = os.path.join(_BASE, "data", "tide", "BAYEX-TG-EXT_RL_ESL_PRED.nc")
OUT_PATH = os.path.join(_BASE, "data", "esl_return_levels_fl.parquet")

# Florida + a small coastal buffer (covers Panhandle through the Keys)
FL_LAT_MIN, FL_LAT_MAX = 24.0, 31.2
FL_LON_MIN, FL_LON_MAX = -87.8, -79.7

METHOD_VARS = {
    "conv": "BAYEX_RL_ESL_CONV_TPXO",
    "mhw":  "BAYEX_RL_ESL_MHW_TPXO",
    "mhhw": "BAYEX_RL_ESL_MHHW_TPXO",
    "hat":  "BAYEX_RL_ESL_HAT_TPXO",
}
STAT_NAMES = ["central", "lower", "upper"]


def main():
    ds = nc.Dataset(NC_PATH)
    lat = np.asarray(ds.variables["BAYEX_LAT_PRED"][:])
    lon = np.asarray(ds.variables["BAYEX_LON_PRED"][:])
    mask = (
        (lat >= FL_LAT_MIN) & (lat <= FL_LAT_MAX) &
        (lon >= FL_LON_MIN) & (lon <= FL_LON_MAX)
    )
    site_idx = np.where(mask)[0]
    n_sites = len(site_idx)
    return_periods = np.arange(2, 1001)  # 999 values, per the file's global attrs

    # arr shape per method: (statistics=3, return_periods=999, n_sites)
    data_by_method = {
        code: np.asarray(ds.variables[varname][:, :, site_idx])
        for code, varname in METHOD_VARS.items()
    }
    ds.close()

    n_periods = len(return_periods)
    site_ids  = site_idx.repeat(n_periods)
    site_lats = lat[site_idx].repeat(n_periods)
    site_lons = lon[site_idx].repeat(n_periods)
    rps       = np.tile(return_periods, n_sites)

    out = {
        "site_id": site_ids,
        "lat": site_lats,
        "lon": site_lons,
        "return_period": rps,
    }
    for code, arr in data_by_method.items():
        for stat_i, stat_name in enumerate(STAT_NAMES):
            # arr[stat_i] shape (return_periods, n_sites) -> flatten to match tile order (site-major, period-minor)
            out[f"esl_{code}_{stat_name}"] = arr[stat_i].T.reshape(-1)

    df = pd.DataFrame(out)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows, {n_sites} FL sites -> {OUT_PATH}")


if __name__ == "__main__":
    main()
