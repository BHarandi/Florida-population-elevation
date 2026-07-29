"""
Florida Population by Elevation — Streamlit Dashboard
Author: Bellah Harandi
Date: July 2026

Run: python -m streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import shape
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from PIL import Image, ImageDraw
import json
import io
import base64
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Florida Population by Elevation",
    page_icon="🌊",
    layout="wide",
)

_BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(_BASE, "data", "population_by_elevation.parquet")
COUNTY_SHP = os.path.join(_BASE, "data", "shp", "counties", "tl_2010_12_county10.shp")
STATE_SHP  = os.path.join(_BASE, "data", "shp", "state",    "tl_2020_12_state.shp")
DEM_PATH      = os.path.join(_BASE, "data", "dem_florida_100m.tif")
_wp_local     = os.path.join(_BASE, "data", "worldpop_wgs84")
WORLDPOP_DIR  = _wp_local if os.path.isdir(_wp_local) else r"E:\2026\Datasets\worldpop-data\wgs84"
_HAZARDS_LOCAL  = os.path.join(_BASE, "data", "Florida_Hazards_1996-2024.parquet")
_HAZARDS_GITHUB = "https://raw.githubusercontent.com/BHarandi/Florida-population-elevation/main/data/Florida_Hazards_1996-2024.parquet"
HAZARDS_PATH    = _HAZARDS_LOCAL if os.path.exists(_HAZARDS_LOCAL) else _HAZARDS_GITHUB

# ── Infrastructure data paths ──────────────────────────────────────────────────
# Primary source: GitHub raw URLs (public — works for everyone).
# Falls back to local copy in data/Transportation/ if present (faster for dev),
# then to F: drive Final Data folder.
_GITHUB_BASE = "https://raw.githubusercontent.com/BHarandi/Florida-population-elevation/main/data/Transportation"
_INFRA_LOCAL = os.path.join(_BASE, "data", "Transportation")
_FINAL_DATA  = r"F:\2026\Datasets\infrastracture\Final Data\Transportation"
_fin_local   = os.path.join(_BASE, "data", "Finance")
_fin_github  = "https://raw.githubusercontent.com/BHarandi/Florida-population-elevation/main/data/Finance"
FINANCE_DIR  = _fin_local if os.path.isdir(_fin_local) else _fin_github
FINANCE_FILES = [
    "F10_grsales_cy1011.xlsx", "F10_grsales_cy1213.xlsx",
    "F10_grsales_cy1415.xlsx", "F10_grsales_cy1617.xlsx",
    "F10_grsales_cy1819.xlsx", "F10_grsales_cy2021.xlsx",
    "F10_grsales_cy2223.xlsx", "F10_grsales_cy2425.xlsx",
]


def _resolve_layer_path(lcfg: dict) -> str:
    """Prefer local copy (fast), else GitHub raw URL (public access)."""
    local = lcfg.get("local", "")
    if local:
        p = os.path.join(_INFRA_LOCAL, local)
        if os.path.exists(p):
            return p
        f = os.path.join(_FINAL_DATA, local)
        if os.path.exists(f):
            return f
        return f"{_GITHUB_BASE}/{local}"
    return ""


INFRA_LAYERS = {
    "Roadways": {
        "local": "Roadways.geojson",
        "color": "#e377c2", "group": "Transportation",
        "is_line": True,
    },
    "Bridges": {
        "local": "Bridges.geojson",
        "color": "#ff7f0e", "group": "Transportation",
        "is_line": True,
    },
    "Bus Terminals": {
        "local": "Bus Terminals.geojson",
        "color": "#d62728", "group": "Transportation",
    },
    "Rail Facilities (Lines)": {
        "local": "Rail Facilities (PL).geojson",
        "color": "#7f7f7f", "group": "Transportation",
        "is_line": True,
    },
    "Rail Facilities (Points)": {
        "local": "Rail Facilities (PT).geojson",
        "color": "#595959", "group": "Transportation",
    },
    "Railway": {
        "local": "Railway.geojson",
        "color": "#2ca02c", "group": "Transportation",
        "is_line": True,
    },
    "Ports": {
        "local": "Ports.geojson",
        "color": "#8c564b", "group": "Transportation",
    },
    "Marinas": {
        "local": "Marinas.geojson",
        "color": "#9467bd", "group": "Transportation",
    },
    "Aviation (Airports)": {
        "local": "Airports.geojson",
        "color": "#1f77b4", "group": "Transportation",
    },
}

BAND_ORDER_M  = ["0-1 m",   "1-2 m",   "2-5 m",   "5-10 m",  "10-25 m", "25-50 m", "50+ m"]
BAND_ORDER_FT = ["0-3 ft",  "3-7 ft",  "7-16 ft", "16-33 ft","33-82 ft","82-164 ft","164+ ft"]

BAND_MAP_M_TO_FT = dict(zip(BAND_ORDER_M, BAND_ORDER_FT))
BAND_MAP_FT_TO_M = dict(zip(BAND_ORDER_FT, BAND_ORDER_M))

BAND_COLORS_M = {
    "0-1 m":   "#4575b4", "1-2 m":   "#1a9850", "2-5 m":   "#66bd63",
    "5-10 m":  "#a6d96a", "10-25 m": "#d4aa4a", "25-50 m": "#a06020",
    "50+ m":   "#6b3a0f",
}
BAND_COLORS_FT = {
    "0-3 ft":   "#4575b4", "3-7 ft":   "#1a9850", "7-16 ft":  "#66bd63",
    "16-33 ft": "#a6d96a", "33-82 ft": "#d4aa4a", "82-164 ft":"#a06020",
    "164+ ft":  "#6b3a0f",
}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March",    4: "April",
    5: "May",     6: "June",     7: "July",      8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}
MONTH_NUM = {v: k for k, v in MONTH_NAMES.items()}



# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_parquet(DATA_PATH)


@st.cache_data(show_spinner="Loading hazards data…")
def load_hazards_data():
    _is_url = HAZARDS_PATH.startswith("http")
    if not _is_url and not os.path.exists(HAZARDS_PATH):
        return None
    cols = [
        "GEOID", "CZ_NAME", "EVENT_TYPE", "HAZARD",
        "start_year", "BEGIN_LAT", "BEGIN_LON",
        "ADJ_DAMAGE_PROPERTY", "TOTAL_DEATHS", "TOTAL_INJURIES",
    ]
    try:
        df = pd.read_parquet(HAZARDS_PATH, columns=cols)
    except Exception:
        return None
    df["start_year"] = df["start_year"].astype(int)
    df["ADJ_DAMAGE_PROPERTY"] = pd.to_numeric(df["ADJ_DAMAGE_PROPERTY"], errors="coerce").fillna(0)
    df["TOTAL_DEATHS"]        = pd.to_numeric(df["TOTAL_DEATHS"],        errors="coerce").fillna(0)
    df["TOTAL_INJURIES"]      = pd.to_numeric(df["TOTAL_INJURIES"],      errors="coerce").fillna(0)
    hazard_name_map = (
        df.dropna(subset=["HAZARD", "EVENT_TYPE"])
        .groupby("HAZARD")["EVENT_TYPE"]
        .agg(lambda x: x.value_counts().index[0])
    )
    df["HAZARD"] = df["HAZARD"].map(hazard_name_map).fillna(df["HAZARD"])
    return df


@st.cache_data
def load_county_geojson():
    """Load Florida county boundaries from local 2010 TIGER shapefile."""
    if not os.path.exists(COUNTY_SHP):
        return None, None
    gdf = gpd.read_file(COUNTY_SHP)                      # already Florida-only (state FIPS 12)
    gdf = gdf[["GEOID10", "NAME10", "geometry"]].copy()
    gdf = gdf.to_crs(epsg=4326)
    return json.loads(gdf.to_json()), gdf[["GEOID10", "NAME10"]]


@st.cache_data
def load_state_boundary():
    """Load Florida state boundary — returns list of (lons, lats) per polygon ring."""
    if not os.path.exists(STATE_SHP):
        return []
    gdf = gpd.read_file(STATE_SHP).to_crs(epsg=4326)
    rings = []
    for geom in gdf.geometry:
        for poly in geom.geoms:                 # iterate MultiPolygon parts
            coords = list(poly.exterior.coords)
            rings.append(([c[0] for c in coords], [c[1] for c in coords]))
    return rings


@st.cache_data
def load_state_geometry_wkt():
    """Return Florida state boundary as a single WGS84 WKT string for DEM clipping."""
    if not os.path.exists(STATE_SHP):
        return None
    from shapely.ops import unary_union
    gdf = gpd.read_file(STATE_SHP).to_crs(epsg=4326)
    return unary_union(gdf.geometry).wkt


@st.cache_data(show_spinner="Loading gross sales data — first run only…")
def load_finance_data():
    """Parse all F10 Excel files → long-format DataFrame. Works local or via GitHub URL."""
    _is_url = FINANCE_DIR.startswith("http")
    SKIP = {'Summary', 'Line Item Detail'}
    records = []
    for fname in FINANCE_FILES:
        xl = None
        actual_path = None
        if not _is_url:
            _lp = os.path.join(FINANCE_DIR, fname)
            if os.path.exists(_lp):
                try:
                    xl = pd.ExcelFile(_lp)
                    actual_path = _lp
                except Exception:
                    pass
        if xl is None:
            _up = f"{_fin_github}/{fname}"
            try:
                xl = pd.ExcelFile(_up)
                actual_path = _up
            except Exception:
                continue
        for sheet in xl.sheet_names:
            if sheet in SKIP:
                continue
            try:
                raw = pd.read_excel(actual_path, sheet_name=sheet, header=None)
            except Exception:
                continue
            hdr_idx = None
            for i, row in raw.iterrows():
                if any(str(v).strip() == 'Kind Code' for v in row.values):
                    hdr_idx = i
                    break
            if hdr_idx is None:
                continue
            hdr = list(raw.iloc[hdr_idx].values)
            month_cols = {}
            for ci in range(2, len(hdr)):
                v = hdr[ci]
                try:
                    if pd.isna(v):
                        continue
                except Exception:
                    pass
                try:
                    dt = pd.to_datetime(v)
                    month_cols[ci] = (dt.year, dt.month)
                except Exception:
                    pass
            if not month_cols:
                continue
            for ri in range(hdr_idx + 1, len(raw)):
                row = list(raw.iloc[ri].values)
                try:
                    kc = int(float(str(row[0]).strip()))
                except (ValueError, TypeError):
                    continue
                kind_name = str(row[1]).strip() if len(row) > 1 else ''
                for ci, (yr, mo) in month_cols.items():
                    if ci >= len(row):
                        continue
                    try:
                        sales = float(row[ci])
                    except (TypeError, ValueError):
                        continue
                    if pd.isna(sales) or sales < 0:
                        continue
                    records.append((sheet, yr, mo, kc, kind_name, sales))
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records, columns=[
        'county', 'year', 'month', 'kind_code', 'kind_name', 'gross_sales'
    ])
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
    )
    return df


@st.cache_data(show_spinner="Loading infrastructure layer…")
def load_infra_layer(path: str, simplify_tol: float = 0.0):
    """Load an infrastructure shapefile or GeoJSON into a WGS-84 GeoDataFrame."""
    if not path or not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            return None, "File loaded but contains no features."
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)
        if simplify_tol > 0:
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.simplify(simplify_tol, preserve_topology=True)
        return gdf, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner="Sampling DEM for elevation profile…")
def _infra_elev_bands(path: str, simplify_tol: float, county_wkt, county_bbox):
    """
    Load an infrastructure layer, clip to county polygon, sample the DEM at each
    point centroid, and return a DataFrame with columns [_band (metric), _elev_m].
    """
    if not path or not os.path.exists(path):
        return None
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            return None
        gdf = gdf.set_crs(epsg=4326) if gdf.crs is None else gdf.to_crs(epsg=4326)
        if simplify_tol > 0:
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.simplify(simplify_tol, preserve_topology=True)
    except Exception:
        return None

    # County clip
    if county_wkt:
        try:
            from shapely import wkt as _swkt
            gdf = gpd.clip(gdf, _swkt.loads(county_wkt))
        except Exception:
            if county_bbox:
                gdf = gdf.cx[county_bbox[0]:county_bbox[2], county_bbox[1]:county_bbox[3]]
    if gdf.empty:
        return None

    # Centroids
    lons = gdf.geometry.apply(lambda g: g.centroid.x if g and not g.is_empty else None)
    lats = gdf.geometry.apply(lambda g: g.centroid.y if g and not g.is_empty else None)
    valid = lons.notna() & lats.notna()
    lons, lats = lons[valid].tolist(), lats[valid].tolist()
    if not lons:
        return None

    # Sample DEM
    if not os.path.exists(DEM_PATH):
        return None
    try:
        with rasterio.open(DEM_PATH) as src:
            nodata = src.nodata
            elevs  = [v[0] for v in src.sample(zip(lons, lats))]
    except Exception:
        return None

    def _band(e):
        if nodata is not None and abs(float(e) - float(nodata)) < 1:
            return None
        e = float(e)
        if e < 0:   return None
        if e < 1:   return "0-1 m"
        if e < 2:   return "1-2 m"
        if e < 5:   return "2-5 m"
        if e < 10:  return "5-10 m"
        if e < 25:  return "10-25 m"
        if e < 50:  return "25-50 m"
        return "50+ m"

    df = pd.DataFrame({"_band": [_band(e) for e in elevs],
                       "_elev_m": [float(e) for e in elevs]})
    df = df.dropna(subset=["_band"])
    return df if not df.empty else None


def _infra_hover_texts(gdf: "gpd.GeoDataFrame") -> list:
    """Build hover label strings for an infrastructure GeoDataFrame."""
    cols = set(gdf.columns)
    name_col   = next((c for c in ["Name", "NAME", "FACILITYNAM", "LABEL", "SITENAME"] if c in cols), None)
    county_col = next((c for c in ["COUNTY", "County", "COUNTY_NAM"] if c in cols), None)
    names    = gdf[name_col].fillna("—").astype(str)   if name_col   else ["—"] * len(gdf)
    counties = gdf[county_col].fillna("").astype(str)  if county_col else [""] * len(gdf)
    return [
        f"<b>{n.strip() or '—'}</b>" + (f"<br>{c.strip()} County" if c.strip() else "")
        for n, c in zip(names, counties)
    ]


@st.cache_data(show_spinner="Reading DEM …")
def get_dem_overlay(geom_wkt: str, unit_k: str):
    """
    Clip the 10 m DEM to a county geometry, colorize with 5 elevation classes,
    and return (data_uri_png, [west, south, east, north], hover_dict).
    Returns (None, None, None) if the DEM is missing or clipping fails.
    """
    if not os.path.exists(DEM_PATH):
        return None, None, None

    try:
        from shapely import wkt as shapely_wkt
        geom_wgs84 = shapely_wkt.loads(geom_wkt)
        gdf = gpd.GeoDataFrame(geometry=[geom_wgs84], crs="EPSG:4326").to_crs("EPSG:4269")
        geom_4269 = gdf.geometry.iloc[0]
    except Exception:
        return None, None, None

    try:
        with rasterio.open(DEM_PATH) as src:
            out_image, out_transform = rio_mask(
                src, [geom_4269.__geo_interface__], crop=True, filled=False,
            )
    except Exception:
        return None, None, None

    from rasterio.features import geometry_mask
    dem_ma = out_image[0]  # numpy masked array: mask=True where outside polygon or DEM nodata

    h, w = dem_ma.shape
    if h == 0 or w == 0:
        return None, None, None

    # Polygon boundary mask (True = outside the county polygon)
    poly_outside = geometry_mask(
        [geom_4269.__geo_interface__],
        out_shape=(h, w),
        transform=out_transform,
        invert=False,
    )
    # Inside-polygon nodata = masked by rasterio AND inside the polygon
    inside_nodata = dem_ma.mask & ~poly_outside

    dem = dem_ma.filled(np.nan).astype(np.float32)

    west  = out_transform.c
    north = out_transform.f
    east  = west  + w * out_transform.a
    south = north + h * out_transform.e

    # Downsample for display — max 600 px per axis
    MAX_PX = 600
    step_h = max(1, h // MAX_PX)
    step_w = max(1, w // MAX_PX)
    dem_ds = dem[::step_h, ::step_w]
    dem_disp = dem_ds * 3.28084 if unit_k == "Feet" else dem_ds
    poly_outside_ds  = poly_outside[::step_h, ::step_w]
    inside_nodata_ds = inside_nodata[::step_h, ::step_w]

    # 5 elevation classes + below-0 water — colours match BAND_COLORS_FT/M
    if unit_k == "Feet":
        bands = [
            (-9999,  0,   ( 33, 102, 172)),   # below 0 ft  — deep blue
            (    0,  3,   ( 69, 117, 180)),   # 0–3 ft      — blue
            (    3,  7,   ( 26, 152,  80)),   # 3–7 ft      — dark green
            (    7, 16,   (102, 189,  99)),   # 7–16 ft     — medium green
            (   16, 33,   (166, 217, 106)),   # 16–33 ft    — light green
            (   33, 82,   (212, 170,  74)),   # 33–82 ft    — tan
            (   82,164,   (160,  96,  32)),   # 82–164 ft   — brown
            (  164,9999,  (107,  58,  15)),   # 164+ ft     — dark brown
        ]
        band_labels = ["below 0 ft","0–3 ft","3–7 ft","7–16 ft","16–33 ft","33–82 ft","82–164 ft","164+ ft"]
        unit_str = "ft"
    else:
        bands = [
            (-9999,  0,   ( 33, 102, 172)),   # below 0 m  — deep blue
            (    0,  1,   ( 69, 117, 180)),   # 0–1 m      — #4575b4
            (    1,  2,   ( 26, 152,  80)),   # 1–2 m      — #1a9850
            (    2,  5,   (102, 189,  99)),   # 2–5 m      — #66bd63
            (    5, 10,   (166, 217, 106)),   # 5–10 m     — #a6d96a
            (   10, 25,   (212, 170,  74)),   # 10–25 m    — #d4aa4a
            (   25, 50,   (160,  96,  32)),   # 25–50 m    — #a06020
            (   50, 9999, (107,  58,  15)),   # 50+ m      — #6b3a0f
        ]
        band_labels = ["below 0 m", "0–1 m", "1–2 m", "2–5 m",
                       "5–10 m", "10–25 m", "25–50 m", "50+ m"]
        unit_str = "m"

    rows, cols = dem_disp.shape
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
    label_arr = np.full((rows, cols), "", dtype=object)
    for (low, high, (r, g, b)), lbl in zip(bands, band_labels):
        px = (dem_disp >= low) & (dem_disp < high)
        rgba[px] = [r, g, b, 205]
        label_arr[px] = lbl
    # Outside polygon → fully transparent
    rgba[poly_outside_ds] = [0, 0, 0, 0]
    # Inside polygon but DEM has no data (bridges, buildings, gaps) → neutral gray
    rgba[inside_nodata_ds] = [160, 160, 160, 140]

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # Hover grid — ~60×60 points across the county
    HOVER_N = 60
    sh = max(1, rows // HOVER_N)
    sw = max(1, cols // HOVER_N)
    hdem   = dem_disp[::sh, ::sw]
    hlabel = label_arr[::sh, ::sw]
    hr, hc = hdem.shape
    lon_arr = np.linspace(west, east,  hc)
    lat_arr = np.linspace(north, south, hr)
    lons_m, lats_m = np.meshgrid(lon_arr, lat_arr)

    valid = ~np.isnan(hdem)
    hover = {
        "lons": lons_m[valid].tolist(),
        "lats": lats_m[valid].tolist(),
        "text": [f"{v:.1f} {unit_str} above MSL" for v in hdem[valid].tolist()],
    }

    return data_uri, [west, south, east, north], hover


@st.cache_data(show_spinner="Computing flood overlay …")
def get_flood_overlay(geom_wkt: str, sea_level_m: float):
    """
    Color pixels with elevation <= sea_level_m as flooded (red).
    Already below 0 m → deep blue. Safe land → transparent.
    Returns (data_uri_png, [west, south, east, north]) or (None, None).
    """
    if not os.path.exists(DEM_PATH):
        return None, None

    try:
        from shapely import wkt as shapely_wkt
        geom_wgs84 = shapely_wkt.loads(geom_wkt)
        gdf = gpd.GeoDataFrame(geometry=[geom_wgs84], crs="EPSG:4326").to_crs("EPSG:4269")
        geom_4269 = gdf.geometry.iloc[0]
    except Exception:
        return None, None

    try:
        with rasterio.open(DEM_PATH) as src:
            out_image, out_transform = rio_mask(
                src, [geom_4269.__geo_interface__], crop=True, filled=False,
            )
    except Exception:
        return None, None

    from rasterio.features import geometry_mask
    dem_ma = out_image[0]
    h, w = dem_ma.shape
    if h == 0 or w == 0:
        return None, None

    poly_outside = geometry_mask(
        [geom_4269.__geo_interface__],
        out_shape=(h, w), transform=out_transform, invert=False,
    )
    dem = dem_ma.filled(np.nan).astype(np.float32)

    west  = out_transform.c
    north = out_transform.f
    east  = west  + w * out_transform.a
    south = north + h * out_transform.e

    MAX_PX = 2000
    step_h = max(1, h // MAX_PX)
    step_w = max(1, w // MAX_PX)
    dem_ds          = dem[::step_h, ::step_w]
    poly_outside_ds = poly_outside[::step_h, ::step_w]
    valid           = ~np.isnan(dem_ds) & ~poly_outside_ds

    rgba = np.zeros((dem_ds.shape[0], dem_ds.shape[1], 4), dtype=np.uint8)
    rgba[valid & (dem_ds < 0)]                             = [ 30, 100, 210, 200]  # blue — already below sea level
    rgba[valid & (dem_ds >= 0) & (dem_ds <= sea_level_m)] = [220,   0,   0, 160]  # vivid red semi-transparent — flooded
    rgba[poly_outside_ds]                                  = [  0,   0,   0,   0]  # transparent outside

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return data_uri, [west, south, east, north]


# ── Continuous colormaps for population overlays ──────────────────────────────
def _apply_colormap(norm_arr: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Interpolate norm_arr values [0,1] through an N×3 color stop array."""
    n = len(colors) - 1
    v = np.clip(norm_arr, 0, 1) * n
    lo = np.floor(v).astype(np.int32).clip(0, n - 1)
    hi = (lo + 1).clip(0, n)
    t  = (v - lo)[..., np.newaxis]
    return (colors[lo] * (1 - t) + colors[hi] * t).astype(np.uint8)

_BLUES = np.array([
    [237, 248, 255],
    [198, 219, 239],
    [107, 174, 214],
    [ 33, 113, 181],
    [  8,  48, 107],
], dtype=np.float32)

_YLORRD = np.array([
    [255, 255, 178],
    [254, 204,  92],
    [253, 141,  60],
    [240,  59,  32],
    [189,   0,  38],
], dtype=np.float32)

_LOG_COUNT_MAX = np.log1p(1000.0)     # ~890 peak in Florida (Miami)
_LOG_DENS_MAX  = np.log1p(100_000.0)  # matching density ceiling


@st.cache_data(show_spinner="Loading population map …")
def get_pop_overlay(geom_wkt: str, year: int):
    """Clip WorldPop raster to geometry; return count image, density image, bounds, hover, error."""
    pop_path = os.path.join(WORLDPOP_DIR, f"pop_{year}_florida.tif")
    if not os.path.exists(pop_path):
        return None, None, None, None, None, f"File not found: {pop_path}"

    if not geom_wkt:
        return None, None, None, None, None, "No geometry provided."
    from shapely import wkt as shapely_wkt
    try:
        geom_wgs84 = shapely_wkt.loads(geom_wkt)
    except Exception as e:
        return None, None, None, None, None, f"Invalid geometry WKT: {e}"

    try:
        with rasterio.open(pop_path) as src:
            pop_nodata = src.nodata
            out_image, out_transform = rio_mask(
                src, [geom_wgs84.__geo_interface__], crop=True, filled=False,
            )
    except Exception as e:
        return None, None, None, None, None, f"Raster processing error: {e}"

    from rasterio.features import geometry_mask
    pop_ma = out_image[0]
    h, w = pop_ma.shape
    if h == 0 or w == 0:
        return None, None, None, None, None, "Empty crop — geometry may not overlap raster extent."

    poly_outside = geometry_mask(
        [geom_wgs84.__geo_interface__],
        out_shape=(h, w), transform=out_transform, invert=False,
    )
    pop = pop_ma.filled(np.nan).astype(np.float32)
    # Convert WorldPop's NoData sentinel to NaN (exact match)
    if pop_nodata is not None:
        pop[pop == np.float32(pop_nodata)] = np.nan
    # Safety: any negative value is impossible for population counts
    pop[pop < 0] = np.nan

    west  = out_transform.c
    north = out_transform.f
    east  = west  + w * out_transform.a
    south = north + h * out_transform.e

    # Use full resolution for small areas (counties); cap at 3000px for large areas (statewide)
    MAX_PX = 3000
    step_h = max(1, h // MAX_PX)
    step_w = max(1, w // MAX_PX)
    pop_ds          = pop[::step_h, ::step_w]
    poly_outside_ds = poly_outside[::step_h, ::step_w]

    rows, cols = pop_ds.shape
    # Color pixels with population > 0 (includes fractional values e.g. 0.3, 0.7 people/pixel)
    # Exactly 0 = no people at all; NaN = water/outside boundary
    valid = ~np.isnan(pop_ds) & (pop_ds > 0)

    # ── 5-class quantile breaks (matching ArcGIS Classify → Quantile, 5 classes) ─
    valid_vals = pop_ds[valid]
    if valid_vals.size >= 5:
        q20, q40, q60, q80 = np.percentile(valid_vals, [20, 40, 60, 80])
    else:
        q20, q40, q60, q80 = 0.5, 3.5, 10.5, 17.5

    # Yellow → orange → bright red (matching ArcGIS "Yellow to Red" ramp, 5 classes)
    _Q5_COLORS = [
        (255, 255,   0, 210),  # Q1 — bright yellow
        (255, 168,   0, 215),  # Q2 — amber
        (255,  98,   0, 220),  # Q3 — orange
        (255,  30,   0, 225),  # Q4 — red-orange
        (255,   0,   0, 230),  # Q5 — bright red
    ]

    # ── Density image: 5-class quantile (same upper-value logic as ArcGIS) ───
    rgba_dens = np.zeros((rows, cols, 4), dtype=np.uint8)
    # Class 1: ≤ q20  (includes all 0-valued land pixels when q20 = 0)
    rgba_dens[valid & (pop_ds <= q20)] = _Q5_COLORS[0]
    # Classes 2–4: (prev_break, curr_break]
    rgba_dens[valid & (pop_ds > q20) & (pop_ds <= q40)] = _Q5_COLORS[1]
    rgba_dens[valid & (pop_ds > q40) & (pop_ds <= q60)] = _Q5_COLORS[2]
    rgba_dens[valid & (pop_ds > q60) & (pop_ds <= q80)] = _Q5_COLORS[3]
    # Class 5: > q80
    rgba_dens[valid & (pop_ds > q80)] = _Q5_COLORS[4]
    rgba_dens[poly_outside_ds] = 0

    buf_dens = io.BytesIO()
    Image.fromarray(rgba_dens, "RGBA").save(buf_dens, format="PNG")
    data_uri_dens = "data:image/png;base64," + base64.b64encode(buf_dens.getvalue()).decode()

    # [min, q20, q40, q60, q80, max] in people/km² for the legend
    max_dens = round(float(np.nanmax(valid_vals)) * 100) if valid.any() else 100
    dens_breaks = [0, round(q20 * 100, 1), round(q40 * 100, 1),
                   round(q60 * 100, 1), round(q80 * 100, 1), max_dens]

    # ── Hover grid — ~60×60 sample points ────────────────────────────────────
    HOVER_N = 60
    sh = max(1, rows // HOVER_N)
    sw = max(1, cols // HOVER_N)
    pop_h = pop_ds[::sh, ::sw]
    hr, hc = pop_h.shape
    lon_arr = np.linspace(west, east,  hc)
    lat_arr = np.linspace(north, south, hr)
    lons_m, lats_m = np.meshgrid(lon_arr, lat_arr)
    valid_h = ~np.isnan(pop_h)
    hover = {
        "lons": lons_m[valid_h].tolist(),
        "lats": lats_m[valid_h].tolist(),
        "text": [
            f"{'< 1 person' if v < 0.01 else f'~{v:.1f} people'} | "
            f"{'< 1' if v * 100 < 1 else f'~{v * 100:.0f}'} people/km²"
            for v in pop_h[valid_h].tolist()
        ],
    }

    return None, data_uri_dens, [west, south, east, north], hover, dens_breaks, None


def _pop_legend_html(breaks: list) -> str:
    """Continuous gradient legend — breaks = [min, q20, q40, q60, q80, max] in people/km²."""
    min_d, max_d = breaks[0], breaks[-1]

    def _fmt(v):
        if v >= 10_000: return f"{v/1_000:.0f}k"
        if v >= 1_000:  return f"{v/1_000:.1f}k"
        return f"{v:.0f}"

    gradient = "linear-gradient(to right, #FFFF00, #FF8800, #FF0000)"
    return (
        '<div style="font-size:0.8rem;line-height:1.6;">'
        'Population density (people/km²):'
        '<br>'
        f'<span style="display:inline-block;width:220px;height:12px;'
        f'background:{gradient};border-radius:2px;"></span>'
        '<br>'
        f'<div style="display:flex;justify-content:space-between;width:220px;font-size:0.75rem;">'
        f'<span>{_fmt(min_d)}</span>'
        f'<span>{_fmt(max_d)}</span>'
        f'</div>'
        '</div>'
    )


def _dem_legend_html(unit_k: str) -> str:
    """Return an HTML colour-strip legend for the DEM overlay (5 classes + water)."""
    if unit_k == "Feet":
        items = [
            ("#2166ac", "below 0 ft"),
            ("#4575b4", "0–3 ft"),
            ("#1a9850", "3–7 ft"),
            ("#66bd63", "7–16 ft"),
            ("#a6d96a", "16–33 ft"),
            ("#d4aa4a", "33–82 ft"),
            ("#a06020", "82–164 ft"),
            ("#6b3a0f", "164+ ft"),
        ]
    else:
        items = [
            ("#2166ac", "below 0 m"),
            ("#4575b4", "0–1 m"),
            ("#1a9850", "1–2 m"),
            ("#66bd63", "2–5 m"),
            ("#a6d96a", "5–10 m"),
            ("#d4aa4a", "10–25 m"),
            ("#a06020", "25–50 m"),
            ("#6b3a0f", "50+ m"),
        ]
    swatches = " ".join(
        f'<span title="{label}" style="display:inline-block;width:14px;height:14px;'
        f'background:{color};border-radius:2px;margin-right:2px;vertical-align:middle;"></span>'
        f'<small style="margin-right:8px;">{label}</small>'
        for color, label in items
    )
    return f'<div style="line-height:2;">{swatches}</div>'


def to_display_bands(df, use_feet):
    """Rename Elev_Band from metric to feet names for display."""
    if use_feet:
        df = df.copy()
        df["Elev_Band"] = df["Elev_Band"].map(BAND_MAP_M_TO_FT).fillna(df["Elev_Band"])
    return df

def to_query_band(band_name, use_feet):
    """Convert a display band name to the metric name stored in the parquet."""
    if use_feet:
        return BAND_MAP_FT_TO_M.get(band_name, band_name)
    return band_name

def to_query_bands(bands, use_feet):
    if use_feet:
        return [BAND_MAP_FT_TO_M.get(b, b) for b in bands]
    return bands


df_all = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Florida Population by Elevation (2010–2025)")
st.caption("Author: Bellah Harandi")
st.caption("Supervisors: Ivan David Haigh  |  Thomas Wahl  |  Christopher Emrich")
st.caption("University of Central Florida (UCF)  |  2026")

if df_all is None:
    st.error(
        f"Data file not found: `{DATA_PATH}`\n\n"
        "Run **`create_sample_data.py`** or **`processing.ipynb`** first."
    )
    st.stop()

fl_geojson, county_meta = load_county_geojson()
state_rings = load_state_boundary()


# ── Filter state (widgets live inside the Distribution tab) ───────────────────
_unit      = st.session_state.get("dist_unit", "Feet (ft)")
use_feet   = _unit == "Feet (ft)"
unit_key   = "Feet" if use_feet else "Metric"
band_order  = BAND_ORDER_FT  if use_feet else BAND_ORDER_M
band_colors = BAND_COLORS_FT if use_feet else BAND_COLORS_M
unit_label  = "elevation above MSL (ft)" if use_feet else "elevation above MSL (m)"

all_years = sorted(df_all["Year"].unique())

county_options = ["Florida (Statewide)"] + sorted(
    df_all[df_all["Scope"] == "County"]["County_Name"].unique()
)

# Reset bands when unit changes
if "dist_bands" in st.session_state:
    stale = [b for b in st.session_state["dist_bands"] if b not in band_order]
    if stale:
        st.session_state["dist_bands"] = band_order

selected_area  = st.session_state.get("dist_county", "Florida (Statewide)")
selected_bands = st.session_state.get("dist_bands",  band_order)


# ── Filter helpers ────────────────────────────────────────────────────────────
def get_area_df(area_name, unit_k, yr_min, yr_max, bands):
    scope = "Statewide" if area_name == "Florida (Statewide)" else "County"
    bands_m = to_query_bands(bands, unit_k == "Feet")
    df = df_all[
        (df_all["Scope"] == scope)  &
        (df_all["Year"]  >= yr_min) &
        (df_all["Year"]  <= yr_max) &
        (df_all["Elev_Band"].isin(bands_m))
    ].copy()
    if scope == "County":
        df = df[df["County_Name"] == area_name]
    df = to_display_bands(df, unit_k == "Feet")
    df["Elev_Band"] = pd.Categorical(df["Elev_Band"], categories=band_order, ordered=True)
    return df.sort_values(["Year", "Elev_Band"])


df_area = get_area_df(selected_area, unit_key,
                      min(all_years), max(all_years), selected_bands)


# Pre-initialize infra checkbox state so it survives tab switches without resetting.
for _ln_pre in INFRA_LAYERS:
    _k_pre = f"infra_{_ln_pre.replace(' ', '_')}"
    if _k_pre not in st.session_state:
        st.session_state[_k_pre] = _ln_pre in ("Aviation (Airports)",)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Distribution", "Map", "Sea Level Rise", "FEMA Lifeline", "Economic Activity", "Hazards"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Distribution (single year snapshot)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    # ── Inline filters ────────────────────────────────────────────────────────
    fi_c1, fi_c2, fi_c3 = st.columns([1, 1, 3])
    with fi_c1:
        st.radio("Elevation unit", ["Feet (ft)", "Metric (m)"],
                 horizontal=True, key="dist_unit")
    with fi_c2:
        st.selectbox("County / Statewide", county_options, key="dist_county")
    with fi_c3:
        st.multiselect("Elevation bands", options=band_order,
                       default=band_order, key="dist_bands")
    st.markdown("---")

    col_ctrl, _ = st.columns([1, 3])
    snap_year   = col_ctrl.selectbox("Select year", all_years,
                                      index=len(all_years) - 1, key="snap_year")
    df_snap = df_area[df_area["Year"] == snap_year].sort_values("Elev_Band")

    if not selected_bands:
        st.info("Select at least one elevation band above.")
    elif df_snap.empty:
        # Check whether any band has data for this county (to give a helpful message)
        scope = "Statewide" if selected_area == "Florida (Statewide)" else "County"
        county_bands_m = df_all[df_all["Scope"] == scope] if scope == "Statewide" else df_all[(df_all["Scope"] == scope) & (df_all["County_Name"] == selected_area)]
        county_bands_m = county_bands_m["Elev_Band"].unique()
        if use_feet:
            available = [BAND_MAP_M_TO_FT.get(b, b) for b in county_bands_m if BAND_MAP_M_TO_FT.get(b, b) in BAND_ORDER_FT]
            available = sorted(available, key=lambda x: BAND_ORDER_FT.index(x) if x in BAND_ORDER_FT else 99)
        else:
            available = [b for b in BAND_ORDER_M if b in county_bands_m]
        if available:
            st.warning(
                f"No population recorded for the selected band(s) in **{selected_area}**. "
                f"Available elevation bands: {', '.join(available)}."
            )
        else:
            st.warning("No data for this selection.")
    else:
        total_pop = df_snap["Population"].sum()
        col_ctrl.metric("Total population", f"{total_pop:,.0f}")
        col_ctrl.metric("Year", snap_year)
        col_ctrl.markdown(
            f'<div style="font-size:0.875rem;color:#6b7280;margin-top:4px;">Area</div>'
            f'<div style="font-size:1.75rem;font-weight:400;line-height:1.2;letter-spacing:-0.01em;">{selected_area}</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            fig_bar = px.bar(
                df_snap, x="Elev_Band", y="Population",
                color="Elev_Band", color_discrete_map=band_colors,
                text="Pct_of_State",
                title=f"Population by elevation ({unit_label}) — {snap_year}",
                labels={"Population": "Population", "Elev_Band": "Elevation band"},
                category_orders={"Elev_Band": band_order},
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            fig_pie = px.pie(
                df_snap, names="Elev_Band", values="Pct_of_State",
                color="Elev_Band", color_discrete_map=band_colors,
                title=f"Population share by elevation — {snap_year}",
                hole=0.4,
                category_orders={"Elev_Band": band_order},
            )
            fig_pie.update_traces(
                hovertemplate="<b>%{label}</b><br>%{value:.2f}% of state population<extra></extra>"
            )
            fig_pie.update_layout(height=420)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.dataframe(
            df_snap[["Elev_Band", "Population", "Pct_of_State"]]
            .rename(columns={"Elev_Band": f"Elevation ({unit_label})",
                              "Pct_of_State": "% of State"})
            .reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Map (Florida county choropleth)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if fl_geojson is None:
        st.warning(f"County shapefile not found at: `{COUNTY_SHP}`")
    else:
        st.subheader("Florida counties — population by elevation band")

        map_col1, map_col2 = st.columns([3, 1])

        with map_col2:
            map_year = st.selectbox("Year", all_years,
                                    index=len(all_years) - 1, key="map_year")

            map_unit      = st.radio("Elevation unit", ["Feet (ft)", "Metric (m)"],
                                     horizontal=True, key="map_unit")
            map_use_feet   = map_unit == "Feet (ft)"
            map_band_order  = BAND_ORDER_FT  if map_use_feet else BAND_ORDER_M
            map_band_colors = BAND_COLORS_FT if map_use_feet else BAND_COLORS_M
            map_unit_label  = "elevation above MSL (ft)"  if map_use_feet else "elevation above MSL (m)"

            # Reset band selection if unit changed
            if "map_band" in st.session_state and st.session_state["map_band"] not in (["All elevations"] + map_band_order):
                st.session_state["map_band"] = "All elevations"

            band_options = ["All elevations"] + map_band_order
            map_band = st.selectbox("Elevation band", band_options, key="map_band")

            map_county_options = ["All counties"] + sorted(
                df_all[df_all["Scope"] == "County"]["County_Name"].unique()
            )
            # Use a non-widget storage key so map clicks can set it without conflict
            if "map_county_sel" not in st.session_state:
                st.session_state["map_county_sel"] = "All counties"
            _mc_idx = (
                map_county_options.index(st.session_state["map_county_sel"])
                if st.session_state["map_county_sel"] in map_county_options else 0
            )
            map_county = st.selectbox("County", map_county_options, index=_mc_idx)
            # Keep storage key in sync when user changes the dropdown manually
            st.session_state["map_county_sel"] = map_county

            map_metric = st.radio("Colour by", ["Population", "% of State"],
                                  horizontal=True)

        # ── Build county data ─────────────────────────────────────────────────
        if map_band == "All elevations":
            df_map = (
                df_all[
                    (df_all["Scope"] == "County") &
                    (df_all["Year"]  == map_year)
                ]
                .groupby(["County_GEOID", "County_Name"], as_index=False)
                .agg(Population=("Population", "sum"))
            )
            state_total = df_map["Population"].sum()
            df_map["Pct_of_State"] = (df_map["Population"] / state_total * 100).round(2)
            band_title = "All elevations"
        else:
            df_map = df_all[
                (df_all["Scope"]     == "County") &
                (df_all["Year"]      == map_year) &
                (df_all["Elev_Band"] == to_query_band(map_band, map_use_feet))
            ][["County_GEOID", "County_Name", "Population", "Pct_of_State"]].copy()
            band_title = map_band

        color_col   = "Population" if map_metric == "Population" else "Pct_of_State"
        color_label = "Population" if map_metric == "Population" else "% of State"

        if df_map.empty:
            st.warning("No data for this selection.")
        else:
            n_counties = df_map["County_GEOID"].nunique()
            with map_col1:
                st.caption(f"{n_counties} counties  |  {band_title}  |  {map_year}")

            highlight_df = pd.DataFrame()
            if map_county != "All counties":
                highlight_df = df_map[df_map["County_Name"] == map_county]

            fig_map = px.choropleth(
                df_map,
                geojson=fl_geojson,
                locations="County_GEOID",
                featureidkey="properties.GEOID10",
                color=color_col,
                hover_name="County_Name",
                hover_data={"Population": ":,.0f", "Pct_of_State": ":.2f",
                             "County_GEOID": False},
                color_continuous_scale="Reds",
                labels={color_col: color_label},
                title=f"Florida — {band_title} ({map_year})",
            )
            if not highlight_df.empty:
                fig_map.add_choropleth(
                    geojson=fl_geojson,
                    locations=highlight_df["County_GEOID"].tolist(),
                    featureidkey="properties.GEOID10",
                    z=[1] * len(highlight_df),
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                    showscale=False,
                    marker=dict(line=dict(color="gold", width=3)),
                    hoverinfo="skip", name="selected",
                )
            for i, (lons, lats) in enumerate(state_rings):
                fig_map.add_scattergeo(
                    lon=lons, lat=lats, mode="lines",
                    line=dict(color="black", width=1.5),
                    showlegend=False, hoverinfo="skip",
                    name=f"_boundary_{i}",
                )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(
                height=650, margin={"r": 0, "t": 40, "l": 0, "b": 0},
                coloraxis_colorbar=dict(title=color_label),
            )

            with map_col1:
                event = st.plotly_chart(fig_map, use_container_width=True,
                                         on_select="rerun", selection_mode="points",
                                         key="county_map")
                if event and event.selection and event.selection.get("points"):
                    clicked_geoid = event.selection["points"][0].get("location")
                    if clicked_geoid:
                        match = df_map[df_map["County_GEOID"] == clicked_geoid]["County_Name"]
                        if not match.empty:
                            clicked_name = match.iloc[0]
                            if (clicked_name in map_county_options and
                                    clicked_name != st.session_state.get("map_county_sel")):
                                st.session_state["map_county_sel"] = clicked_name
                                st.rerun()

        # ── Detail table — full width below map ───────────────────────────────
        detail_label = map_county if map_county != "All counties" else "Florida (Statewide)"
        det_scope = "Statewide" if map_county == "All counties" else "County"
        detail = df_all[
            (df_all["Scope"] == det_scope) &
            (df_all["Year"]  == map_year)
        ].copy()
        detail = to_display_bands(detail, map_use_feet)
        if map_county != "All counties":
            detail = detail[detail["County_Name"] == map_county]

        detail["Elev_Band"] = pd.Categorical(
            detail["Elev_Band"], categories=map_band_order, ordered=True)
        detail = detail.sort_values("Elev_Band")

        band_col = f"Band ({map_unit_label})"
        detail_display = (
            detail[["Elev_Band", "Population", "Pct_of_State"]]
            .rename(columns={"Elev_Band": band_col, "Pct_of_State": "% State"})
            .reset_index(drop=True)
        )

        st.markdown(f"**{detail_label} — population by elevation band ({map_year})**")

        def highlight_band(row):
            if map_band != "All elevations" and row[band_col] == map_band:
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)

        st.dataframe(
            detail_display.style.apply(highlight_band, axis=1),
            use_container_width=True, hide_index=True,
        )
        st.caption(f"Total population: {detail['Population'].sum():,.0f}")

        # ══════════════════════════════════════════════════════════════════════
        # COUNTY ZOOM & ELEVATION PROFILE — shown only when a county is selected
        # ══════════════════════════════════════════════════════════════════════
        if map_county != "All counties" and map_county in df_map["County_Name"].values:
            st.markdown("---")
            zoom_col1, zoom_col3 = st.columns(2)

            # ── Get county geometry + centroid ────────────────────────────────
            county_geoid_sel = df_map[
                df_map["County_Name"] == map_county
            ]["County_GEOID"].iloc[0]

            county_feat_list = [
                f for f in fl_geojson["features"]
                if f["properties"]["GEOID10"] == county_geoid_sel
            ]
            county_geojson_single = {"type": "FeatureCollection",
                                      "features": county_feat_list}

            if county_feat_list:
                geom       = shape(county_feat_list[0]["geometry"])
                center_lat = geom.centroid.y
                center_lon = geom.centroid.x
                minx, miny, maxx, maxy = geom.bounds
                max_span   = max(maxx - minx, maxy - miny)
                zoom_level = max(6, min(10, round(8.0 - max_span * 6)))
            else:
                geom = None
                center_lat, center_lon, zoom_level = 27.5, -81.5, 7

            # ── Zoom map with DEM overlay ─────────────────────────────────────
            with zoom_col1:
                st.markdown(f"**{map_county} — elevation map (DEM)**")

                if geom is None:
                    st.info("County geometry not found.")
                    dem_img = dem_bounds = dem_hover = None
                else:
                    dem_img, dem_bounds, dem_hover = get_dem_overlay(geom.wkt, "Feet" if map_use_feet else "Metric")

                # Basemap + DEM layer controls
                _basemap_map = {
                    "Streets (OpenStreetMap)": "open-street-map",
                    "Light (Carto)":           "carto-positron",
                    "Dark (Carto)":            "carto-darkmatter",
                }
                ctrl_sel, ctrl_bmap, ctrl_dem = st.columns([2, 1, 1])
                basemap_style = ctrl_sel.selectbox(
                    "Basemap style",
                    options=list(_basemap_map.keys()),
                    index=0,
                    key="basemap_style",
                    label_visibility="collapsed",
                )
                show_basemap = ctrl_bmap.toggle("Basemap", value=True, key="show_basemap")
                show_dem     = ctrl_dem.toggle("DEM",     value=True, key="show_dem")

                mapbox_style = _basemap_map[basemap_style] if show_basemap else "white-bg"
                dem_opacity  = 0.78 if show_basemap else 1.0

                # Build county boundary lons/lats for outline trace
                if geom is not None and geom.geom_type == "MultiPolygon":
                    boundary_lons, boundary_lats = [], []
                    for poly in geom.geoms:
                        coords = list(poly.exterior.coords)
                        boundary_lons += [c[0] for c in coords] + [None]
                        boundary_lats += [c[1] for c in coords] + [None]
                elif geom is not None:
                    coords = list(geom.exterior.coords)
                    boundary_lons = [c[0] for c in coords]
                    boundary_lats = [c[1] for c in coords]
                else:
                    boundary_lons, boundary_lats = [], []

                fig_zoom = go.Figure()
                fig_zoom.add_trace(go.Scattermapbox(
                    lon=boundary_lons,
                    lat=boundary_lats,
                    mode="lines",
                    line=dict(color="black", width=2.5),
                    hoverinfo="skip",
                    showlegend=False,
                ))

                mapbox_cfg = dict(
                    style=mapbox_style,
                    zoom=zoom_level,
                    center={"lat": center_lat, "lon": center_lon},
                )
                if dem_img is not None and show_dem:
                    w84, s84, e84, n84 = dem_bounds
                    mapbox_cfg["layers"] = [{
                        "sourcetype": "image",
                        "source": dem_img,
                        "coordinates": [
                            [w84, n84],
                            [e84, n84],
                            [e84, s84],
                            [w84, s84],
                        ],
                        "opacity": dem_opacity,
                        "below": "traces",
                    }]

                # Invisible hover-grid — lets user see elevation on mouse-over
                if dem_hover is not None and show_dem:
                    fig_zoom.add_trace(go.Scattermapbox(
                        lon=dem_hover["lons"],
                        lat=dem_hover["lats"],
                        mode="markers",
                        marker=dict(size=14, color="rgba(0,0,0,0)"),
                        text=dem_hover["text"],
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=False,
                        name="",
                    ))

                fig_zoom.update_layout(
                    mapbox=mapbox_cfg,
                    height=440,
                    margin={"r": 0, "t": 10, "l": 0, "b": 0},
                    uirevision=map_county,  # preserve user zoom/pan unless county changes
                )
                st.plotly_chart(fig_zoom, use_container_width=True, config={"scrollZoom": True})

                if dem_img is not None and show_dem:
                    st.markdown(_dem_legend_html("Feet" if map_use_feet else "Metric"), unsafe_allow_html=True)
                elif dem_img is None and geom is not None:
                    st.warning("DEM file not found — outline only.")

            # ── Right column: population density map (yellow→red) ────────────
            if geom is not None:
                pop_img_count, pop_img_dens, pop_bounds, pop_hover, pop_dens_breaks, _pop_err = get_pop_overlay(geom.wkt, map_year)
            else:
                pop_img_count, pop_img_dens, pop_bounds, pop_hover, pop_dens_breaks, _pop_err = None, None, None, None, None, "No county geometry."
            _pop_bmap_map = {
                "Streets (OpenStreetMap)": "open-street-map",
                "Light (Carto)":           "carto-positron",
                "Dark (Carto)":            "carto-darkmatter",
            }
            with zoom_col3:
                st.markdown(f"**{map_county} — population density ({map_year})**")
                pd_sel, pd_bmap, pd_tog = st.columns([2, 1, 1])
                pop_dens_bstyle  = pd_sel.selectbox("Basemap", options=list(_pop_bmap_map.keys()), index=0, key="pop_dens_basemap_county", label_visibility="collapsed")
                show_dens_bmap   = pd_bmap.toggle("Basemap",  value=True, key="pop_dens_show_basemap_county")
                show_dens        = pd_tog.toggle("Population density",   value=True, key="show_dens_county")
                pop_dens_style   = _pop_bmap_map[pop_dens_bstyle] if show_dens_bmap else "white-bg"
                if pop_img_dens is None:
                    st.info(_pop_err or f"WorldPop raster for {map_year} not available.")
                else:
                    fig_dens = go.Figure()
                    fig_dens.add_trace(go.Scattermapbox(
                        lon=boundary_lons, lat=boundary_lats, mode="lines",
                        line=dict(color="black", width=2.5),
                        hoverinfo="skip", showlegend=False,
                    ))
                    if pop_hover:
                        fig_dens.add_trace(go.Scattermapbox(
                            lon=pop_hover["lons"], lat=pop_hover["lats"],
                            mode="markers",
                            marker=dict(size=14, color="rgba(0,0,0,0)"),
                            text=pop_hover["text"],
                            hovertemplate="%{text}<extra></extra>",
                            showlegend=False, name="",
                        ))
                    pw84, ps84, pe84, pn84 = pop_bounds
                    _dens_layers = [{
                        "sourcetype": "image",
                        "source": pop_img_dens,
                        "coordinates": [
                            [pw84, pn84], [pe84, pn84],
                            [pe84, ps84], [pw84, ps84],
                        ],
                        "opacity": 0.85,
                        "below": "traces",
                    }] if show_dens else []
                    fig_dens.update_layout(
                        mapbox=dict(
                            style=pop_dens_style,
                            zoom=zoom_level,
                            center={"lat": center_lat, "lon": center_lon},
                            layers=_dens_layers,
                        ),
                        height=440,
                        margin={"r": 0, "t": 10, "l": 0, "b": 0},
                        uirevision=f"{map_county}_pop_dens",
                    )
                    st.plotly_chart(fig_dens, use_container_width=True, config={"scrollZoom": True})
                    if show_dens and pop_dens_breaks:
                        st.markdown(_pop_legend_html(pop_dens_breaks), unsafe_allow_html=True)

            # ── Elevation profile chart (below, full width) ───────────────────
            st.markdown(f"**{map_county} — elevation profile ({map_year})**")

            elev_profile = df_all[
                (df_all["Scope"]       == "County") &
                (df_all["Year"]        == map_year) &
                (df_all["County_Name"] == map_county)
            ].copy()
            elev_profile = to_display_bands(elev_profile, map_use_feet)
            elev_profile["Elev_Band"] = pd.Categorical(
                elev_profile["Elev_Band"], categories=map_band_order, ordered=True)
            elev_profile = elev_profile.sort_values("Elev_Band")

            fig_profile = go.Figure()
            for _, row in elev_profile.iterrows():
                color = map_band_colors.get(row["Elev_Band"], "#888888")
                fig_profile.add_trace(go.Bar(
                    x=[row["Elev_Band"]],
                    y=[row["Population"]],
                    marker_color=color,
                    marker_line_color="white",
                    marker_line_width=1.5,
                    name=str(row["Elev_Band"]),
                    hovertemplate=(
                        f"<b>{row['Elev_Band']}</b><br>"
                        f"Population: {row['Population']:,}<br>"
                        f"% of State: {row['Pct_of_State']:.2f}%<extra></extra>"
                    ),
                ))

            fig_profile.add_trace(go.Scatter(
                x=elev_profile["Elev_Band"].tolist(),
                y=elev_profile["Population"].tolist(),
                mode="lines",
                line=dict(color="rgba(60,60,60,0.6)", width=2, shape="spline"),
                fill="tozeroy",
                fillcolor="rgba(100,149,237,0.12)",
                showlegend=False,
                hoverinfo="skip",
            ))

            fig_profile.update_layout(
                title=f"Population by elevation — {map_county}",
                xaxis_title=f"Elevation ({map_unit_label})",
                yaxis_title="Population",
                showlegend=False,
                height=400,
                margin={"r": 10, "t": 50, "l": 10, "b": 50},
                plot_bgcolor="#f8f9fa",
                xaxis=dict(categoryorder="array", categoryarray=map_band_order),
            )
            _, _chart_mid, _ = st.columns([1, 2, 1])
            with _chart_mid:
                st.plotly_chart(fig_profile, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # STATEWIDE DEM — shown when no county is selected
        # ══════════════════════════════════════════════════════════════════════
        elif map_county == "All counties":
            st.markdown("---")
            state_col1, state_col3 = st.columns(2)

            # ── Statewide DEM map ─────────────────────────────────────────────
            with state_col1:
                st.markdown("**Florida — elevation map (DEM)**")
                state_wkt = load_state_geometry_wkt()
                if state_wkt:
                    dem_img, dem_bounds, dem_hover = get_dem_overlay(state_wkt, "Feet" if map_use_feet else "Metric")

                    _basemap_map_state = {
                        "Streets (OpenStreetMap)": "open-street-map",
                        "Light (Carto)":           "carto-positron",
                        "Dark (Carto)":            "carto-darkmatter",
                    }
                    s_sel, s_bmap, s_dem = st.columns([2, 1, 1])
                    state_basemap_style = s_sel.selectbox(
                        "Basemap style", options=list(_basemap_map_state.keys()),
                        index=0, key="state_basemap_style", label_visibility="collapsed",
                    )
                    show_state_basemap = s_bmap.toggle("Basemap", value=True, key="state_show_basemap")
                    show_state_dem     = s_dem.toggle("DEM",     value=True, key="state_show_dem")

                    mapbox_style_state = _basemap_map_state[state_basemap_style] if show_state_basemap else "white-bg"
                    dem_opacity_state  = 0.78 if show_state_basemap else 1.0

                    fig_state = go.Figure()
                    for lons, lats in state_rings:
                        fig_state.add_trace(go.Scattermapbox(
                            lon=lons, lat=lats, mode="lines",
                            line=dict(color="black", width=2),
                            hoverinfo="skip", showlegend=False,
                        ))

                    mapbox_cfg_state = dict(
                        style=mapbox_style_state,
                        zoom=5.5,
                        center={"lat": 27.8, "lon": -81.5},
                    )
                    if dem_img is not None and show_state_dem:
                        w84, s84, e84, n84 = dem_bounds
                        mapbox_cfg_state["layers"] = [{
                            "sourcetype": "image",
                            "source": dem_img,
                            "coordinates": [
                                [w84, n84], [e84, n84], [e84, s84], [w84, s84],
                            ],
                            "opacity": dem_opacity_state,
                            "below": "traces",
                        }]

                    if dem_hover is not None and show_state_dem:
                        fig_state.add_trace(go.Scattermapbox(
                            lon=dem_hover["lons"], lat=dem_hover["lats"],
                            mode="markers",
                            marker=dict(size=14, color="rgba(0,0,0,0)"),
                            text=dem_hover["text"],
                            hovertemplate="%{text}<extra></extra>",
                            showlegend=False, name="",
                        ))

                    fig_state.update_layout(
                        mapbox=mapbox_cfg_state,
                        height=480,
                        margin={"r": 0, "t": 10, "l": 0, "b": 0},
                        uirevision="state_dem",
                    )
                    st.plotly_chart(fig_state, use_container_width=True, config={"scrollZoom": True})

                    if dem_img is not None and show_state_dem:
                        st.markdown(_dem_legend_html("Feet" if map_use_feet else "Metric"), unsafe_allow_html=True)
                    elif dem_img is None:
                        st.warning("DEM file not found — outline only.")

            # ── Right column: statewide population density ────────────────────
            _pop_bmap_map_s = {
                "Streets (OpenStreetMap)": "open-street-map",
                "Light (Carto)":           "carto-positron",
                "Dark (Carto)":            "carto-darkmatter",
            }
            if state_wkt:
                pop_img_count_s, pop_img_dens_s, pop_bounds_s, pop_hover_s, pop_dens_breaks_s, _pop_err_s = get_pop_overlay(state_wkt, map_year)
            else:
                pop_img_count_s, pop_img_dens_s, pop_bounds_s, pop_hover_s, pop_dens_breaks_s, _pop_err_s = None, None, None, None, None, "State boundary shapefile not found."

            with state_col3:
                st.markdown(f"**Florida — population density ({map_year})**")
                pds_sel, pds_bmap, pds_tog = st.columns([2, 1, 1])
                pop_dens_bstyle_s  = pds_sel.selectbox("Basemap", options=list(_pop_bmap_map_s.keys()), index=0, key="pop_dens_basemap_state", label_visibility="collapsed")
                show_dens_bmap_s   = pds_bmap.toggle("Basemap",  value=True, key="pop_dens_show_basemap_state")
                show_dens_s        = pds_tog.toggle("Population density",   value=True, key="show_dens_state")
                pop_dens_style_s   = _pop_bmap_map_s[pop_dens_bstyle_s] if show_dens_bmap_s else "white-bg"
                if pop_img_dens_s is None:
                    st.info(_pop_err_s or f"WorldPop raster for {map_year} not available.")
                else:
                    fig_dens_s = go.Figure()
                    for lons, lats in state_rings:
                        fig_dens_s.add_trace(go.Scattermapbox(
                            lon=lons, lat=lats, mode="lines",
                            line=dict(color="black", width=2),
                            hoverinfo="skip", showlegend=False,
                        ))
                    if pop_hover_s:
                        fig_dens_s.add_trace(go.Scattermapbox(
                            lon=pop_hover_s["lons"], lat=pop_hover_s["lats"],
                            mode="markers",
                            marker=dict(size=14, color="rgba(0,0,0,0)"),
                            text=pop_hover_s["text"],
                            hovertemplate="%{text}<extra></extra>",
                            showlegend=False, name="",
                        ))
                    pw84s, ps84s, pe84s, pn84s = pop_bounds_s
                    _dens_layers_s = [{
                        "sourcetype": "image",
                        "source": pop_img_dens_s,
                        "coordinates": [
                            [pw84s, pn84s], [pe84s, pn84s],
                            [pe84s, ps84s], [pw84s, ps84s],
                        ],
                        "opacity": 0.85,
                        "below": "traces",
                    }] if show_dens_s else []
                    fig_dens_s.update_layout(
                        mapbox=dict(
                            style=pop_dens_style_s,
                            zoom=5.5,
                            center={"lat": 27.8, "lon": -81.5},
                            layers=_dens_layers_s,
                        ),
                        height=480,
                        margin={"r": 0, "t": 10, "l": 0, "b": 0},
                        uirevision="state_pop_dens",
                    )
                    st.plotly_chart(fig_dens_s, use_container_width=True, config={"scrollZoom": True})
                    if show_dens_s and pop_dens_breaks_s:
                        st.markdown(_pop_legend_html(pop_dens_breaks_s), unsafe_allow_html=True)

            # ── Statewide elevation profile chart (below, full width) ─────────
            st.markdown(f"**Florida — elevation profile ({map_year})**")

            elev_profile_state = df_all[
                (df_all["Scope"] == "Statewide") &
                (df_all["Year"]  == map_year)
            ].copy()
            elev_profile_state = to_display_bands(elev_profile_state, map_use_feet)
            elev_profile_state["Elev_Band"] = pd.Categorical(
                elev_profile_state["Elev_Band"], categories=map_band_order, ordered=True)
            elev_profile_state = elev_profile_state.sort_values("Elev_Band")

            fig_state_profile = go.Figure()
            for _, row in elev_profile_state.iterrows():
                color = map_band_colors.get(row["Elev_Band"], "#888888")
                fig_state_profile.add_trace(go.Bar(
                    x=[row["Elev_Band"]],
                    y=[row["Population"]],
                    marker_color=color,
                    marker_line_color="white",
                    marker_line_width=1.5,
                    name=str(row["Elev_Band"]),
                    hovertemplate=(
                        f"<b>{row['Elev_Band']}</b><br>"
                        f"Population: {row['Population']:,}<br>"
                        f"% of State: {row['Pct_of_State']:.2f}%<extra></extra>"
                    ),
                ))

            fig_state_profile.add_trace(go.Scatter(
                x=elev_profile_state["Elev_Band"].tolist(),
                y=elev_profile_state["Population"].tolist(),
                mode="lines",
                line=dict(color="rgba(60,60,60,0.6)", width=2, shape="spline"),
                fill="tozeroy",
                fillcolor="rgba(100,149,237,0.12)",
                showlegend=False,
                hoverinfo="skip",
            ))

            fig_state_profile.update_layout(
                title=f"Population by elevation — Florida ({map_year})",
                xaxis_title=f"Elevation ({map_unit_label})",
                yaxis_title="Population",
                showlegend=False,
                height=400,
                margin={"r": 10, "t": 50, "l": 10, "b": 50},
                plot_bgcolor="#f8f9fa",
                xaxis=dict(categoryorder="array", categoryarray=map_band_order),
            )
            _, _state_chart_mid, _ = st.columns([1, 2, 1])
            with _state_chart_mid:
                st.plotly_chart(fig_state_profile, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # DOWNLOAD SECTION
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### Download data")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        # 1. Selected county — all years
        with dl_col1:
            if map_county != "All counties":
                dl_county = df_all[
                    (df_all["Scope"]       == "County") &
                    (df_all["County_Name"] == map_county)
                ][["Year", "County_GEOID", "County_Name",
                   "Elev_Band", "Elev_Min_m", "Elev_Max_m",
                   "Population", "Pct_of_State"]].sort_values(["Year", "Elev_Min_m"])
                dl_county = to_display_bands(dl_county, use_feet)
                dl_county = dl_county.rename(columns={"County_GEOID": "GEOID"})
                fname = f"{map_county.replace(' ', '_').replace('.', '')}_elevation_{unit_key.lower()}_2010_2025.csv"
                st.download_button(
                    label=f"County: {map_county} (all years)",
                    data=dl_county.to_csv(index=False).encode("utf-8"),
                    file_name=fname,
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("Select a county to enable county download.")

        # 2. All counties — selected year & band
        with dl_col2:
            dl_year_band = df_all[
                (df_all["Scope"] == "County") &
                (df_all["Year"]  == map_year)
            ]
            if map_band != "All elevations":
                dl_year_band = dl_year_band[dl_year_band["Elev_Band"] == to_query_band(map_band, use_feet)]
            dl_year_band = dl_year_band[
                ["County_GEOID", "County_Name", "Elev_Band",
                 "Elev_Min_m", "Elev_Max_m", "Population", "Pct_of_State"]
            ].sort_values(["County_Name", "Elev_Min_m"])
            dl_year_band = to_display_bands(dl_year_band, use_feet)
            dl_year_band = dl_year_band.rename(columns={"County_GEOID": "GEOID"})
            band_slug = band_title.replace(" ", "_").replace(">", "gt").replace("/", "-")
            st.download_button(
                label=f"All counties — {map_year} / {band_title}",
                data=dl_year_band.to_csv(index=False).encode("utf-8"),
                file_name=f"florida_all_counties_{map_year}_{band_slug}_{unit_key.lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # 3. Full dataset
        with dl_col3:
            dl_full = df_all[
                ["Year", "Scope", "County_GEOID", "County_Name",
                 "Elev_Band", "Elev_Min_m", "Elev_Max_m",
                 "Population", "Pct_of_State"]
            ].sort_values(["Year", "County_Name", "Elev_Min_m"])
            dl_full = to_display_bands(dl_full, use_feet)
            dl_full = dl_full.rename(columns={"County_GEOID": "GEOID", "Scope": "LEVEL"})
            st.download_button(
                label=f"Full dataset ({unit_key}, 2010–2025)",
                data=dl_full.to_csv(index=False).encode("utf-8"),
                file_name=f"florida_population_by_elevation_{unit_key.lower()}_2010_2025.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Sea Level Rise
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Sea Level Rise — Flood Risk")

    slr_col1, slr_col2 = st.columns([3, 1])

    with slr_col2:
        slr_area = st.selectbox(
            "County / Statewide", county_options, key="slr_area",
        )
        slr_year = st.selectbox(
            "Year", all_years, index=len(all_years) - 1, key="slr_year",
        )

        # Read unit toggle first (default Feet) so slider range is correct
        slr_use_meters = st.session_state.get("slr_unit_toggle", False)
        slr_use_feet   = not slr_use_meters

        if slr_use_feet:
            slr_ft    = st.slider("Sea level rise (ft)", 0.0, 60.0, 1.0, 0.5, key="slr_slider")
            slr_m     = slr_ft / 3.28084
            slr_label = f"{slr_ft:.1f} ft"
            slr_band_order = BAND_ORDER_FT
            slr_unit_label = "elevation above MSL (ft)"
        else:
            slr_m     = st.slider("Sea level rise (m)", 0.0, 60.0, 0.3, 0.1, key="slr_slider")
            slr_label = f"{slr_m:.1f} m"
            slr_band_order = BAND_ORDER_M
            slr_unit_label = "elevation above MSL (m)"

        # Unit toggle — below the slider
        u_left, u_mid, u_right = st.columns([2, 1, 2])
        u_left.markdown("<div style='text-align:right;padding-top:6px;font-size:0.9rem;'>Feet</div>", unsafe_allow_html=True)
        u_mid.toggle("", value=slr_use_meters, key="slr_unit_toggle", label_visibility="collapsed")
        u_right.markdown("<div style='padding-top:6px;font-size:0.9rem;'>Meters</div>", unsafe_allow_html=True)

        _slr_basemap_map = {
            "Streets (OpenStreetMap)": "open-street-map",
            "Light (Carto)":           "carto-positron",
            "Dark (Carto)":            "carto-darkmatter",
        }
        slr_basemap_style = st.selectbox(
            "Basemap", options=list(_slr_basemap_map.keys()), index=0, key="slr_basemap",
        )

    # ── Get geometry ─────────────────────────────────────────────────────────
    if slr_area == "Florida (Statewide)":
        slr_geom_wkt = load_state_geometry_wkt()
        slr_center   = {"lat": 27.8, "lon": -81.5}
        slr_zoom     = 5.5
    else:
        slr_geoid = df_all[
            (df_all["Scope"] == "County") &
            (df_all["County_Name"] == slr_area)
        ]["County_GEOID"].iloc[0] if not df_all[
            (df_all["Scope"] == "County") &
            (df_all["County_Name"] == slr_area)
        ].empty else None

        slr_feat = [f for f in fl_geojson["features"]
                    if f["properties"]["GEOID10"] == slr_geoid] if (slr_geoid and fl_geojson) else []
        if slr_feat:
            slr_geom     = shape(slr_feat[0]["geometry"])
            slr_geom_wkt = slr_geom.wkt
            slr_center   = {"lat": slr_geom.centroid.y, "lon": slr_geom.centroid.x}
            minx, miny, maxx, maxy = slr_geom.bounds
            max_span = max(maxx - minx, maxy - miny)
            slr_zoom = max(6, min(10, round(8.0 - max_span * 6)))
        else:
            slr_geom_wkt = None

    # ── Flood map ─────────────────────────────────────────────────────────────
    with slr_col1:
        if slr_geom_wkt is None:
            st.warning("Could not load geometry for selected area.")
        else:
            flood_img, flood_bounds = get_flood_overlay(slr_geom_wkt, slr_m)

            fig_slr = go.Figure()
            # Dummy trace — forces Plotly to render as mapbox instead of cartesian
            fig_slr.add_trace(go.Scattermapbox(
                lon=[], lat=[], mode="markers",
                showlegend=False, hoverinfo="skip",
            ))
            # State/county boundary outline
            for lons, lats in state_rings:
                fig_slr.add_trace(go.Scattermapbox(
                    lon=lons, lat=lats, mode="lines",
                    line=dict(color="black", width=1.5),
                    hoverinfo="skip", showlegend=False,
                ))
            mapbox_cfg_slr = dict(
                style=_slr_basemap_map[slr_basemap_style],
                zoom=slr_zoom,
                center=slr_center,
            )
            if flood_img is not None:
                w84, s84, e84, n84 = flood_bounds
                mapbox_cfg_slr["layers"] = [{
                    "sourcetype": "image",
                    "source": flood_img,
                    "coordinates": [
                        [w84, n84], [e84, n84], [e84, s84], [w84, s84],
                    ],
                    "opacity": 0.85,
                    "below": "traces",
                }]
            elif flood_img is None and not os.path.exists(DEM_PATH):
                st.warning("DEM file not found — flood overlay unavailable.")

            fig_slr.update_layout(
                mapbox=mapbox_cfg_slr,
                height=520,
                margin={"r": 0, "t": 10, "l": 0, "b": 0},
                uirevision=f"{slr_area}_{slr_m}",
            )
            st.plotly_chart(fig_slr, use_container_width=True, config={"scrollZoom": True})

            # Legend
            st.markdown(
                '<span style="display:inline-block;width:14px;height:14px;background:#DC0000;'
                'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>'
                f'<small>Flooded at +{slr_label} sea level rise</small>&nbsp;&nbsp;&nbsp;'
                '<span style="display:inline-block;width:14px;height:14px;background:#2166ac;'
                'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>'
                '<small>Already below sea level</small>',
                unsafe_allow_html=True,
            )

    # ── Population at risk from parquet ───────────────────────────────────────
    st.markdown("---")
    st.markdown(f"**Population at risk — {slr_area} ({slr_year}) at +{slr_label} sea level rise**")

    scope_slr = "Statewide" if slr_area == "Florida (Statewide)" else "County"
    at_risk_df = df_all[
        (df_all["Scope"] == scope_slr) &
        (df_all["Year"]  == slr_year)
    ].copy()
    if scope_slr == "County":
        at_risk_df = at_risk_df[at_risk_df["County_Name"] == slr_area]

    at_risk_df["at_risk"] = at_risk_df["Elev_Max_m"] <= slr_m
    at_risk_pop   = at_risk_df[at_risk_df["at_risk"]]["Population"].sum()
    total_pop_slr = at_risk_df["Population"].sum()
    pct_at_risk   = (at_risk_pop / total_pop_slr * 100) if total_pop_slr > 0 else 0

    r1, r2, r3 = st.columns(3)
    r1.metric("Population at risk", f"{at_risk_pop:,.0f}")
    r2.metric("Total population",   f"{total_pop_slr:,.0f}")
    r3.metric("% at risk",          f"{pct_at_risk:.1f}%")

    at_risk_display = to_display_bands(at_risk_df.copy(), slr_use_feet)
    at_risk_display["Elev_Band"] = pd.Categorical(
        at_risk_display["Elev_Band"], categories=slr_band_order, ordered=True)
    at_risk_display = at_risk_display.sort_values("Elev_Band")
    at_risk_display["Status"] = at_risk_display["at_risk"].map(
        {True: "At risk", False: "Safe"})
    st.dataframe(
        at_risk_display[["Elev_Band", "Population", "Pct_of_State", "Status"]]
        .rename(columns={"Elev_Band": f"Elevation ({slr_unit_label})", "Pct_of_State": "% State"})
        .reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Infrastructure
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Florida Infrastructure Layers")

    infra_map_col, infra_ctrl_col = st.columns([3, 1])

    with infra_ctrl_col:
        infra_area = st.selectbox("County / Statewide", county_options, key="infra_area")
        _infra_bmap_opts = {
            "Streets (OpenStreetMap)": "open-street-map",
            "Light (Carto)":           "carto-positron",
            "Dark (Carto)":            "carto-darkmatter",
        }
        _infra_bmap_keys = list(_infra_bmap_opts.keys())
        _infra_bmap_default = _infra_bmap_keys.index("Streets (OpenStreetMap)")
        infra_bmap_style = st.selectbox(
            "Basemap", _infra_bmap_keys,
            index=_infra_bmap_default, key="infra_bmap",
        )
        show_infra_dem = st.toggle("Elevation (DEM)", value=True, key="infra_show_dem")
        infra_dem_unit = st.radio("DEM unit", ["Feet (ft)", "Metric (m)"],
                                  horizontal=True, key="infra_dem_unit")
        _infra_dem_unit_k = "Feet" if infra_dem_unit == "Feet (ft)" else "Metric"

        st.markdown("---")

        # Layer checkboxes grouped by category
        _layer_groups = {}
        for _ln, _lcfg in INFRA_LAYERS.items():
            _layer_groups.setdefault(_lcfg["group"], []).append(_ln)

        active_infra_layers = []
        for _grp, _lnames in _layer_groups.items():
            _expanded = _grp in ("Transportation", "Emergency Facilities")
            with st.expander(f"**{_grp}**", expanded=_expanded):
                for _ln in _lnames:
                    _lcfg = INFRA_LAYERS[_ln]
                    _checked = st.checkbox(
                        _ln,
                        key=f"infra_{_ln.replace(' ', '_')}",
                    )
                    if _checked:
                        active_infra_layers.append(_ln)

    with infra_map_col:
        # Map viewport
        if infra_area == "Florida (Statewide)":
            _ic = {"lat": 27.8, "lon": -81.5}
            _iz = 5.5
            _cf = None
            _cb = None
            _ifeats = []
        else:
            _igid = df_all[
                (df_all["Scope"] == "County") & (df_all["County_Name"] == infra_area)
            ]["County_GEOID"]
            _igid = _igid.iloc[0] if not _igid.empty else None
            _ifeats = [f for f in fl_geojson["features"]
                       if f["properties"]["GEOID10"] == _igid] if (_igid and fl_geojson) else []
            if _ifeats:
                _igeom = shape(_ifeats[0]["geometry"])
                _ic = {"lat": _igeom.centroid.y, "lon": _igeom.centroid.x}
                _ibx, _iby, _ibxx, _ibyy = _igeom.bounds
                _iz = max(6, min(10, round(8.0 - max(_ibxx - _ibx, _ibyy - _iby) * 6)))
                _cf = infra_area
                _cb = (_ibx, _iby, _ibxx, _ibyy)
            else:
                _ic = {"lat": 27.8, "lon": -81.5}
                _iz = 5.5
                _cf = infra_area
                _cb = None

        # DEM overlay for selected area
        _infra_dem_img = _infra_dem_bounds = None
        if show_infra_dem and os.path.exists(DEM_PATH):
            _dem_wkt_infra = None
            if infra_area == "Florida (Statewide)":
                _dem_wkt_infra = load_state_geometry_wkt()
            elif _ifeats:
                _dem_wkt_infra = _igeom.wkt
            if _dem_wkt_infra:
                try:
                    _infra_dem_img, _infra_dem_bounds, _ = get_dem_overlay(_dem_wkt_infra, _infra_dem_unit_k)
                except Exception:
                    _infra_dem_img = _infra_dem_bounds = None

        fig_infra = go.Figure()

        # State boundary outline
        for _bl, _bla in state_rings:
            fig_infra.add_trace(go.Scattermapbox(
                lon=_bl, lat=_bla, mode="lines",
                line=dict(color="black", width=1),
                hoverinfo="skip", showlegend=False,
            ))

        # County boundary highlight (thicker gold outline)
        if infra_area != "Florida (Statewide)" and _ifeats:
            if _igeom.geom_type == "MultiPolygon":
                _cb_lons, _cb_lats = [], []
                for _poly in _igeom.geoms:
                    _cc = list(_poly.exterior.coords)
                    _cb_lons += [c[0] for c in _cc] + [None]
                    _cb_lats += [c[1] for c in _cc] + [None]
            else:
                _cc = list(_igeom.exterior.coords)
                _cb_lons = [c[0] for c in _cc]
                _cb_lats = [c[1] for c in _cc]
            fig_infra.add_trace(go.Scattermapbox(
                lon=_cb_lons, lat=_cb_lats, mode="lines",
                line=dict(color="gold", width=3),
                hoverinfo="skip", showlegend=False,
            ))

        _summary_rows = []

        for _ln in active_infra_layers:
            _lcfg  = INFRA_LAYERS[_ln]
            _simp  = _lcfg.get("simplify", 0.0)
            _gdf, _err = load_infra_layer(_resolve_layer_path(_lcfg), simplify_tol=_simp)

            if _gdf is None:
                if _err and not _lcfg.get("optional"):
                    st.warning(f"**{_ln}**: {_err}")
                continue
            if _gdf.empty:
                continue

            # County filter — clip to exact county polygon, fall back to bbox
            _fgdf = _gdf
            if _cf:
                if _ifeats:
                    try:
                        _fgdf = gpd.clip(_gdf, _igeom)
                    except Exception:
                        _fgdf = _gdf.cx[_cb[0]:_cb[2], _cb[1]:_cb[3]] if _cb else _gdf
                elif _cb:
                    _fgdf = _gdf.cx[_cb[0]:_cb[2], _cb[1]:_cb[3]]

            if _fgdf.empty:
                _summary_rows.append({"Layer": _ln, "Features": 0})
                continue

            if _lcfg.get("is_line"):
                # Line geometry → render as Scattermapbox lines
                _all_lons, _all_lats = [], []
                for _geom in _fgdf.geometry:
                    if _geom is None or _geom.is_empty:
                        continue
                    _segs = _geom.geoms if _geom.geom_type.startswith("Multi") else [_geom]
                    for _seg in _segs:
                        try:
                            _coords = list(_seg.coords)
                            _all_lons.extend([c[0] for c in _coords] + [None])
                            _all_lats.extend([c[1] for c in _coords] + [None])
                        except Exception:
                            pass
                if _all_lons:
                    fig_infra.add_trace(go.Scattermapbox(
                        lon=_all_lons, lat=_all_lats, mode="lines",
                        line=dict(color=_lcfg["color"], width=1.5),
                        name=_ln,
                        showlegend=True, hoverinfo="skip",
                    ))
                _summary_rows.append({"Layer": _ln, "Features": len(_fgdf)})

            else:
                # Point / polygon → use centroid for marker position
                _pts = _fgdf.copy()
                _pts["_lon"] = _pts.geometry.apply(lambda g: g.centroid.x if g and not g.is_empty else None)
                _pts["_lat"] = _pts.geometry.apply(lambda g: g.centroid.y if g and not g.is_empty else None)
                _pts = _pts.dropna(subset=["_lon", "_lat"])
                if _pts.empty:
                    continue

                _htexts = _infra_hover_texts(_pts)
                fig_infra.add_trace(go.Scattermapbox(
                    lon=_pts["_lon"].tolist(),
                    lat=_pts["_lat"].tolist(),
                    mode="markers",
                    marker=dict(size=8, color=_lcfg["color"], opacity=0.85),
                    text=_htexts,
                    hovertemplate="%{text}<extra></extra>",
                    name=_ln,
                    showlegend=True,
                ))
                _summary_rows.append({"Layer": _ln, "Features": len(_pts)})

        _infra_mapbox_layers = []
        if _infra_dem_img is not None:
            _dw, _ds, _de, _dn = _infra_dem_bounds
            _infra_mapbox_layers = [{
                "sourcetype": "image",
                "source": _infra_dem_img,
                "coordinates": [[_dw, _dn], [_de, _dn], [_de, _ds], [_dw, _ds]],
                "opacity": 0.65,
                "below": "traces",
            }]

        fig_infra.update_layout(
            mapbox=dict(
                style=_infra_bmap_opts[infra_bmap_style],
                zoom=_iz,
                center=_ic,
                layers=_infra_mapbox_layers,
            ),
            height=680,
            margin={"r": 0, "t": 10, "l": 0, "b": 0},
            legend=dict(
                yanchor="top", y=0.98, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.82)",
                font=dict(size=12),
            ),
            uirevision=f"infra_{infra_area}",
        )
        st.plotly_chart(fig_infra, use_container_width=True, config={"scrollZoom": True})

        if _summary_rows:
            st.markdown(f"**Visible layers — feature counts ({infra_area})**")
            st.dataframe(
                pd.DataFrame(_summary_rows),
                use_container_width=False, hide_index=True,
            )
        elif not active_infra_layers:
            st.info("Select one or more layers from the panel on the right to display them on the map.")

        if _infra_dem_img is not None:
            st.markdown(_dem_legend_html(_infra_dem_unit_k), unsafe_allow_html=True)

        # Note about optional layers that need local data
        _missing_optional = [
            _ln
            for _ln in active_infra_layers
            if INFRA_LAYERS[_ln].get("optional") and not os.path.exists(INFRA_LAYERS[_ln]["path"])
        ]
        if _missing_optional:
            st.caption(
                f"⚠ Not available without local data: {', '.join(_missing_optional)}. "
                "These layers require the E: drive dataset."
            )

    # ── Elevation profile — full-width below map columns ───────────────────────
    # Include ALL active layers (line layers use feature centroids for elevation)
    if active_infra_layers and os.path.exists(DEM_PATH):
        st.markdown("---")
        st.subheader("Features by Elevation Band")

        _use_ft     = _infra_dem_unit_k == "Feet"
        _e_band_ord = BAND_ORDER_FT  if _use_ft else BAND_ORDER_M
        _e_band_col = BAND_COLORS_FT if _use_ft else BAND_COLORS_M
        _e_axis_lbl = "elevation above MSL (ft)" if _use_ft else "elevation above MSL (m)"

        _bbox_tuple = tuple(_cb) if _cb else None
        _county_wkt_elev = _igeom.wkt if _ifeats else None
        _elev_rows  = []

        for _ln in active_infra_layers:
            _lcfg = INFRA_LAYERS[_ln]
            _edf  = _infra_elev_bands(_resolve_layer_path(_lcfg), _lcfg.get("simplify", 0.0),
                                      _county_wkt_elev, _bbox_tuple)
            if _edf is None or _edf.empty:
                continue
            _edf = _edf.copy()
            if _use_ft:
                _edf["_band"] = _edf["_band"].map(BAND_MAP_M_TO_FT)
            for _bnd, _cnt in _edf["_band"].value_counts().items():
                _elev_rows.append({"Layer": _ln, "Elev_Band": _bnd,
                                   "Count": int(_cnt), "_color": _lcfg["color"]})

        if _elev_rows:
            _elev_df    = pd.DataFrame(_elev_rows)
            _lyr_colors = {r["Layer"]: r["_color"] for r in _elev_rows}
            _elev_df["Elev_Band"] = pd.Categorical(
                _elev_df["Elev_Band"], categories=_e_band_ord, ordered=True)
            _elev_df = _elev_df.sort_values("Elev_Band")
            _unique_layers = list(dict.fromkeys(r["Layer"] for r in _elev_rows))

            if len(_unique_layers) == 1:
                # Single layer — color bars by elevation band
                fig_elev = px.bar(
                    _elev_df, x="Elev_Band", y="Count",
                    color="Elev_Band", color_discrete_map=_e_band_col,
                    labels={"Elev_Band": _e_axis_lbl, "Count": "Feature count"},
                    category_orders={"Elev_Band": _e_band_ord},
                )
                fig_elev.update_layout(height=400, showlegend=False,
                                       margin={"t": 20, "b": 10})
                st.plotly_chart(fig_elev, use_container_width=True)
            else:
                # Multiple layers — grouped bars, one color per layer
                fig_elev = px.bar(
                    _elev_df, x="Elev_Band", y="Count",
                    color="Layer", color_discrete_map=_lyr_colors,
                    barmode="group",
                    labels={"Elev_Band": _e_axis_lbl, "Count": "Feature count"},
                    category_orders={"Elev_Band": _e_band_ord},
                )
                fig_elev.update_layout(
                    height=420, margin={"t": 20, "b": 10},
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                )
                st.plotly_chart(fig_elev, use_container_width=True)

            # Summary table
            _pivot = _elev_df.pivot_table(
                index="Layer", columns="Elev_Band",
                values="Count", aggfunc="sum", fill_value=0,
            )
            _pivot.columns.name = None
            _pivot.index.name   = "Layer"
            _pivot = _pivot.reindex(columns=[b for b in _e_band_ord if b in _pivot.columns])
            _pivot["Total"] = _pivot.sum(axis=1)
            st.dataframe(_pivot, use_container_width=True)
        else:
            st.info("No elevation data available for the selected layers.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Economic Activity (Florida F10 Gross Sales)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Florida Gross Sales Activity (2010–2025)")

    fin_df = load_finance_data()

    if fin_df.empty:
        st.warning(f"Finance data not found at: `{FINANCE_DIR}`")
    else:
        # Pre-build county list so map clicks can update the selectbox via session state
        fin_county_opts = ["Florida (Statewide)"] + sorted(
            c for c in fin_df["county"].unique() if c != "Statewide"
        )
        if "fin_area_sel" not in st.session_state:
            st.session_state["fin_area_sel"] = "Florida (Statewide)"

        fin_ctrl_col, fin_map_col = st.columns([1, 3])

        with fin_ctrl_col:
            # Area — index driven by session state so map clicks update the dropdown
            _fa_idx = (
                fin_county_opts.index(st.session_state["fin_area_sel"])
                if st.session_state["fin_area_sel"] in fin_county_opts else 0
            )
            fin_area = st.selectbox(
                "County / Statewide  *(click map to select)*",
                fin_county_opts, index=_fa_idx, key="fin_area_dd",
            )
            st.session_state["fin_area_sel"] = fin_area

            # Year
            fin_years = sorted(fin_df["year"].unique().tolist())
            fin_year  = st.selectbox("Year", ["All Years"] + fin_years, key="fin_year")

            # Month
            fin_month = st.selectbox(
                "Month",
                ["All Months"] + list(MONTH_NAMES.values()),
                key="fin_month",
            )

            # Kind Code — single select, only codes with data for current year/area
            _kc_src = fin_df.copy()
            if fin_year != "All Years":
                _kc_src = _kc_src[_kc_src["year"] == int(fin_year)]
            if fin_area != "Florida (Statewide)":
                _kc_src = _kc_src[_kc_src["county"] == fin_area]
            kc_ref = (
                _kc_src.sort_values("date")
                .drop_duplicates(subset=["kind_code"], keep="last")
                [["kind_code", "kind_name"]]
                .sort_values("kind_code")
            )
            kc_options = [
                f"{int(r.kind_code)} — {r.kind_name}" for _, r in kc_ref.iterrows()
            ]
            _kc_default_idx = next(
                (i + 1 for i, k in enumerate(kc_options) if k.startswith("111 ")), 0
            )
            fin_kc_sel = st.selectbox(
                "Kind Code (business type)",
                ["All Kind Codes"] + kc_options,
                index=_kc_default_idx,
                key="fin_kc",
            )
            selected_kc = (
                int(fin_kc_sel.split(" — ")[0]) if fin_kc_sel != "All Kind Codes" else None
            )

            fin_line_color = "#9d174d"  # dark pink, matches map palette

        # ── Choropleth map ────────────────────────────────────────────────────
        map_df = fin_df[fin_df["county"] != "Statewide"].copy()
        if fin_year != "All Years":
            map_df = map_df[map_df["year"] == int(fin_year)]
        if fin_month != "All Months":
            map_df = map_df[map_df["month"] == MONTH_NUM[fin_month]]
        if selected_kc is not None:
            map_df = map_df[map_df["kind_code"] == selected_kc]

        county_sales = map_df.groupby("county", as_index=False)["gross_sales"].sum()

        # Join to GEOID10 for choropleth
        if county_meta is not None:
            _name_geoid = dict(zip(county_meta["NAME10"], county_meta["GEOID10"]))
            county_sales["GEOID"] = county_sales["county"].map(_name_geoid)
            county_sales = county_sales.dropna(subset=["GEOID"])

        # Scale to millions for readable colorbar tick labels
        county_sales["gross_sales_M"] = county_sales["gross_sales"] / 1e6

        yr_lbl  = str(fin_year)
        mo_lbl  = fin_month
        kc_lbl  = fin_kc_sel

        with fin_map_col:
            if county_sales.empty or "GEOID" not in county_sales.columns:
                st.info("No map data for this selection.")
            else:
                fig_fin = px.choropleth(
                    county_sales,
                    geojson=fl_geojson,
                    locations="GEOID",
                    featureidkey="properties.GEOID10",
                    color="gross_sales_M",
                    hover_name="county",
                    hover_data={
                        "gross_sales": ":$,.0f",
                        "gross_sales_M": False,
                        "GEOID": False,
                    },
                    color_continuous_scale=[
                        [0.00, "#fde8f0"],
                        [0.18, "#f9c0d8"],
                        [0.38, "#d5e9c0"],
                        [0.55, "#8bc34a"],
                        [0.68, "#f06292"],
                        [0.84, "#e91e8c"],
                        [1.00, "#880e4f"],
                    ],
                    labels={
                        "gross_sales_M": "Gross Sales ($M)",
                        "gross_sales": "Gross Sales",
                    },
                    title=f"Gross Sales by County — {yr_lbl} / {mo_lbl} / {kc_lbl}",
                )
                for _i, (_lons, _lats) in enumerate(state_rings):
                    fig_fin.add_scattergeo(
                        lon=_lons, lat=_lats, mode="lines",
                        line=dict(color="black", width=1.5),
                        showlegend=False, hoverinfo="skip",
                        name=f"_fin_boundary_{_i}",
                    )
                # Gold boundary for selected county
                if fin_area != "Florida (Statewide)" and county_meta is not None:
                    _sel_geoid_list = county_sales[county_sales["county"] == fin_area]["GEOID"].tolist()
                    if _sel_geoid_list:
                        fig_fin.add_choropleth(
                            geojson=fl_geojson,
                            locations=_sel_geoid_list,
                            featureidkey="properties.GEOID10",
                            z=[1],
                            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                            showscale=False,
                            marker=dict(line=dict(color="gold", width=3)),
                            hoverinfo="skip",
                            name="selected",
                        )
                fig_fin.update_geos(fitbounds="locations", visible=False)
                fig_fin.update_layout(
                    height=500,
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    coloraxis_colorbar=dict(
                        title="Gross Sales<br>($M = millions)",
                        tickprefix="$",
                        ticksuffix="M",
                        tickformat=",.0f",
                    ),
                )
                _fin_event = st.plotly_chart(
                    fig_fin, use_container_width=True,
                    on_select="rerun", selection_mode="points",
                    key="fin_choropleth",
                )

                # Handle map click → update county dropdown
                if (_fin_event and _fin_event.selection
                        and _fin_event.selection.get("points")
                        and county_meta is not None):
                    _clicked_loc = _fin_event.selection["points"][0].get("location")
                    if _clicked_loc:
                        _geoid_to_name = dict(zip(county_meta["GEOID10"], county_meta["NAME10"]))
                        _clicked_name = _geoid_to_name.get(_clicked_loc, "")
                        if (_clicked_name
                                and _clicked_name in fin_county_opts
                                and _clicked_name != st.session_state.get("fin_area_sel")):
                            st.session_state["fin_area_sel"] = _clicked_name
                            st.rerun()

        # ── Time series chart ─────────────────────────────────────────────────
        st.markdown("---")

        if fin_area == "Florida (Statewide)":
            ts_df = fin_df[fin_df["county"] == "Statewide"].copy()
        else:
            ts_df = fin_df[fin_df["county"] == fin_area].copy()

        # Apply all active filters to the time series
        if fin_year != "All Years":
            ts_df = ts_df[ts_df["year"] == int(fin_year)]
        if fin_month != "All Months":
            ts_df = ts_df[ts_df["month"] == MONTH_NUM[fin_month]]
        if selected_kc is not None:
            ts_df = ts_df[ts_df["kind_code"] == selected_kc]

        ts_title = f"Monthly Gross Sales — {fin_area}"
        if fin_year != "All Years":
            ts_title += f"  |  {fin_year}"
        if fin_month != "All Months":
            ts_title += f"  |  {fin_month}"
        if selected_kc is not None:
            _kc_display = fin_kc_sel.split(" — ", 1)[-1] if " — " in fin_kc_sel else fin_kc_sel
            ts_title += f"  |  {_kc_display}"

        if ts_df.empty:
            st.info("No time series data for this selection.")
        else:
            # Aggregate all selected months/kind code into one total line per date
            ts_agg = ts_df.groupby("date", as_index=False)["gross_sales"].sum()
            # Insert NaN for missing months so Plotly shows gaps instead of connecting lines
            if not ts_agg.empty and fin_month == "All Months":
                _full_range = pd.DataFrame({
                    "date": pd.date_range(ts_agg["date"].min(), ts_agg["date"].max(), freq="MS")
                })
                ts_agg = _full_range.merge(ts_agg, on="date", how="left")
            # Scale to millions so y-axis shows readable numbers (avoids Plotly's "G" for billions)
            ts_agg["gross_sales_M"] = ts_agg["gross_sales"] / 1e6
            fig_ts = px.line(
                ts_agg.sort_values("date"),
                x="date", y="gross_sales_M",
                title=ts_title,
                labels={"date": "Date", "gross_sales_M": "Gross Sales ($M)"},
            )
            fig_ts.update_traces(
                line_color=fin_line_color, line_width=2,
                mode="lines+markers",
                marker=dict(size=4, color=fin_line_color),
                connectgaps=False,
            )

            # When a single year is selected show every month; otherwise every 6 months
            if fin_year != "All Years":
                fig_ts.update_xaxes(dtick="M1", tickformat="%b %Y", tickangle=-45)
            else:
                fig_ts.update_xaxes(dtick="M6", tickformat="%b %Y", tickangle=-45)
            fig_ts.update_yaxes(tickformat="$,.0f", ticksuffix="M")
            fig_ts.update_xaxes(title_text="Date")
            fig_ts.update_layout(
                height=420,
                plot_bgcolor="#f8f9fa",
                hovermode="closest",
                margin={"t": 60, "b": 40, "l": 80, "r": 20},
            )
            st.plotly_chart(fig_ts, use_container_width=True)


            # Annual summary table
            ann = (
                ts_df.groupby("year", as_index=False)["gross_sales"]
                .sum()
                .rename(columns={"year": "Year", "gross_sales": "Gross Sales ($)"})
                [["Year", "Gross Sales ($)"]]
            )
            st.markdown(f"**Annual totals — {fin_area}**")
            st.dataframe(
                ann.set_index("Year").style.format({"Gross Sales ($)": "${:,.0f}"}),
                use_container_width=False, hide_index=False,
            )

            # ── Download section ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("**Download**")
            _dl1, _dl2, _dl3 = st.columns(3)

            _mo_slug   = fin_month.replace(" ", "") if fin_month != "All Months" else "AllMonths"
            _area_slug = fin_area.replace(" ", "_")

            # ── 1. CSV — raw data ─────────────────────────────────────────────
            _csv_ts = ts_df[["county", "year", "month", "kind_code", "kind_name",
                              "gross_sales", "date"]].copy()
            _csv_ts = _csv_ts.rename(columns={
                "date": "Date", "county": "County", "year": "Year",
                "month": "Month", "kind_code": "Kind Code",
                "kind_name": "Kind Name", "gross_sales": "Gross Sales ($)",
            })
            _dl1.download_button(
                label="Download data (CSV)",
                data=_csv_ts.to_csv(index=False).encode("utf-8"),
                file_name=f"gross_sales_{_area_slug}_{fin_year}_{_mo_slug}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # ── 2. HTML Report — map + chart + table (no kaleido needed) ─────
            # Embeds interactive Plotly figures; opens in any browser
            _fig_map_html = ""
            try:
                _fig_map_html = fig_fin.to_html(
                    include_plotlyjs="cdn", full_html=False,
                    config={"responsive": True},
                )
            except Exception:
                _fig_map_html = "<p><em>Map not available for this selection.</em></p>"

            _fig_ts_html = fig_ts.to_html(
                include_plotlyjs=False, full_html=False,
                config={"responsive": True},
            )
            _ann_html = ann.to_html(index=True, border=0, classes="dt")

            _html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Florida Gross Sales — {fin_area} {fin_year}</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:1400px;margin:0 auto;padding:24px;color:#222}}
  h1{{color:#880e4f;margin-bottom:4px}}
  .meta{{background:#fde8f0;border-radius:6px;padding:12px 20px;margin:12px 0 24px 0;font-size:.95rem;line-height:1.9}}
  h2{{color:#880e4f;margin-top:36px;border-bottom:2px solid #fde8f0;padding-bottom:6px}}
  .dt{{border-collapse:collapse;font-size:.9rem;margin-top:8px}}
  .dt th,.dt td{{border:1px solid #ddd;padding:6px 16px;text-align:right}}
  .dt th{{background:#fde8f0;color:#880e4f}}
  footer{{margin-top:40px;color:#999;font-size:.8rem;border-top:1px solid #eee;padding-top:12px}}
</style>
</head>
<body>
<h1>Florida Gross Sales Report</h1>
<div class="meta">
  <b>Area:</b> {fin_area} &nbsp;&nbsp;
  <b>Year:</b> {fin_year} &nbsp;&nbsp;
  <b>Month(s):</b> {mo_lbl} &nbsp;&nbsp;
  <b>Kind Code:</b> {kc_lbl}
</div>
<h2>Gross Sales Map by County</h2>
{_fig_map_html}
<h2>Monthly Gross Sales Chart</h2>
{_fig_ts_html}
<h2>Annual Totals</h2>
{_ann_html}
<footer>University of Central Florida (UCF) &nbsp;|&nbsp; Florida Gross Sales Activity 2010–2025 &nbsp;|&nbsp; Author: Bellah Harandi</footer>
</body>
</html>"""

            _dl2.download_button(
                label="Download report (HTML)",
                data=_html_report.encode("utf-8"),
                file_name=f"gross_sales_report_{_area_slug}_{fin_year}_{_mo_slug}.html",
                mime="text/html",
                use_container_width=True,
                help="Opens in any browser — includes interactive map, chart, and table. No extra software needed.",
            )

            # ── 3. PDF Report (map + chart + table, requires kaleido) ───────────
            try:
                _W = 1400
                # Metadata header image
                _hdr = Image.new("RGB", (_W, 160), "#fde8f0")
                _d = ImageDraw.Draw(_hdr)
                _d.text((30, 18),  "Florida Gross Sales Report",                      fill="#880e4f")
                _d.text((30, 55),  f"Area: {fin_area}     Year: {fin_year}     Month(s): {mo_lbl}", fill="#333333")
                _d.text((30, 85),  f"Kind Code: {kc_lbl}",                            fill="#333333")
                _d.line([(30, 118), (_W - 30, 118)], fill="#e0b0c0", width=1)
                _d.text((30, 126), "University of Central Florida (UCF)  |  Author: Bellah Harandi  |  2026", fill="#888888")

                _map_pil = Image.open(io.BytesIO(
                    fig_fin.to_image(format="png", width=_W, height=700, scale=1)
                )).convert("RGB")
                _ts_pil = Image.open(io.BytesIO(
                    fig_ts.to_image(format="png", width=_W, height=480, scale=1)
                )).convert("RGB")

                # Annual totals table image
                _tbl_rows = [("Year", "Gross Sales")] + [
                    (str(int(r["Year"])), f"${r['Gross Sales ($)']:,.0f}")
                    for _, r in ann.iterrows()
                ]
                _row_h, _tbl_pad = 28, 48
                _tbl_h = _tbl_pad + len(_tbl_rows) * _row_h
                _tbl_img = Image.new("RGB", (_W, _tbl_h), "white")
                _td = ImageDraw.Draw(_tbl_img)
                _td.text((30, 12), "Annual Totals", fill="#880e4f")
                for _ri, (_yr_v, _gs_v) in enumerate(_tbl_rows):
                    _y = _tbl_pad + _ri * _row_h
                    _bg = "#fde8f0" if _ri == 0 else ("#f9f0f4" if _ri % 2 == 0 else "white")
                    _td.rectangle([(0, _y), (_W, _y + _row_h - 1)], fill=_bg)
                    _clr = "#880e4f" if _ri == 0 else "#222222"
                    _td.text((40, _y + 6), _yr_v, fill=_clr)
                    _td.text((200, _y + 6), _gs_v, fill=_clr)

                # Stack all images vertically into one PDF page
                _total_h = 160 + _map_pil.height + _ts_pil.height + _tbl_h
                _canvas = Image.new("RGB", (_W, _total_h), "white")
                _canvas.paste(_hdr,    (0, 0))
                _canvas.paste(_map_pil,(0, 160))
                _canvas.paste(_ts_pil, (0, 160 + _map_pil.height))
                _canvas.paste(_tbl_img,(0, 160 + _map_pil.height + _ts_pil.height))

                _pdf_buf = io.BytesIO()
                _canvas.save(_pdf_buf, format="PDF")
                _dl3.download_button(
                    label="Download report (PDF)",
                    data=_pdf_buf.getvalue(),
                    file_name=f"gross_sales_report_{_area_slug}_{fin_year}_{_mo_slug}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help="PDF with map, chart, and annual totals. Requires kaleido.",
                )
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Hazards (NCEI Storm Events 1996–2024, Florida)
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Florida Hazard Events — NCEI Storm Database (1996–2024)")

    df_hz = load_hazards_data()
    if df_hz is None:
        st.error(f"Hazards file not found:\n`{HAZARDS_PATH}`")
    else:
        # ── Sidebar-style filters (top row) ───────────────────────────────────
        hz_f1, hz_f2, hz_f3, hz_f4 = st.columns([1, 1, 2, 2])

        hz_yr_min, hz_yr_max = int(df_hz["start_year"].min()), int(df_hz["start_year"].max())
        with hz_f1:
            hz_year_range = st.slider(
                "Year range", hz_yr_min, hz_yr_max,
                (hz_yr_min, hz_yr_max), key="hz_year_range",
            )
        with hz_f2:
            hz_metric = st.radio(
                "Measure", ["Events", "Property Damage ($)", "Deaths", "Injuries"],
                key="hz_metric",
            )
        with hz_f3:
            all_hazards = sorted(df_hz["HAZARD"].dropna().unique())
            hz_hazards = st.multiselect(
                "Hazard type", all_hazards, default=all_hazards, key="hz_hazards",
            )
        with hz_f4:
            hz_county_opts = ["All counties"] + sorted(df_hz["CZ_NAME"].dropna().unique())
            hz_county = st.selectbox("County / Zone", hz_county_opts, key="hz_county")

        st.markdown("---")

        # ── Apply filters ─────────────────────────────────────────────────────
        dff = df_hz[
            (df_hz["start_year"] >= hz_year_range[0]) &
            (df_hz["start_year"] <= hz_year_range[1]) &
            (df_hz["HAZARD"].isin(hz_hazards))
        ].copy()
        if hz_county != "All counties":
            dff = dff[dff["CZ_NAME"] == hz_county]

        metric_col = {
            "Events":               None,
            "Property Damage ($)":  "ADJ_DAMAGE_PROPERTY",
            "Deaths":               "TOTAL_DEATHS",
            "Injuries":             "TOTAL_INJURIES",
        }[hz_metric]

        # ── KPI metrics ───────────────────────────────────────────────────────
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total events",          f"{len(dff):,}")
        kpi2.metric("Total property damage", f"${dff['ADJ_DAMAGE_PROPERTY'].sum()/1e6:.1f} M")
        kpi3.metric("Total deaths",          f"{int(dff['TOTAL_DEATHS'].sum()):,}")
        kpi4.metric("Total injuries",        f"{int(dff['TOTAL_INJURIES'].sum()):,}")

        st.markdown("---")
        hz_col1, hz_col2 = st.columns(2)

        # ── Bar chart — top hazard types ──────────────────────────────────────
        with hz_col1:
            if metric_col is None:
                bar_df = (dff.groupby("EVENT_TYPE")
                            .size()
                            .reset_index(name="value")
                            .sort_values("value", ascending=False)
                            .head(15))
                bar_label = "Number of events"
            else:
                bar_df = (dff.groupby("EVENT_TYPE")[metric_col]
                            .sum()
                            .reset_index(name="value")
                            .sort_values("value", ascending=False)
                            .head(15))
                bar_label = hz_metric

            fig_hz_bar = px.bar(
                bar_df, x="value", y="EVENT_TYPE",
                orientation="h",
                labels={"value": bar_label, "EVENT_TYPE": "Event type"},
                title=f"Top event types — {hz_metric}",
                color="value",
                color_continuous_scale="Reds",
            )
            fig_hz_bar.update_layout(
                height=440, showlegend=False,
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_hz_bar, use_container_width=True)

        # ── Time series — by year ─────────────────────────────────────────────
        with hz_col2:
            if metric_col is None:
                ts_df = (dff.groupby(["start_year", "HAZARD"])
                           .size()
                           .reset_index(name="value"))
                ts_label = "Number of events"
            else:
                ts_df = (dff.groupby(["start_year", "HAZARD"])[metric_col]
                           .sum()
                           .reset_index(name="value"))
                ts_label = hz_metric

            fig_hz_ts = px.line(
                ts_df, x="start_year", y="value", color="HAZARD",
                labels={"start_year": "Year", "value": ts_label, "HAZARD": "Hazard"},
                title=f"{hz_metric} per year by hazard type",
                markers=True,
            )
            fig_hz_ts.update_layout(height=440, legend=dict(font=dict(size=10)))
            st.plotly_chart(fig_hz_ts, use_container_width=True)

        # ── County breakdown ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Hazards by county**")

        geo_json_hz, county_df_hz = load_county_geojson()

        if metric_col is None:
            county_agg = dff.groupby("GEOID").size().reset_index(name="value")
            top_county_agg = dff.groupby("CZ_NAME").size().reset_index(name="value")
            choro_label = "Events"
        else:
            county_agg = dff.groupby("GEOID")[metric_col].sum().reset_index(name="value")
            top_county_agg = dff.groupby("CZ_NAME")[metric_col].sum().reset_index(name="value")
            choro_label = hz_metric

        if county_df_hz is not None:
            county_agg = county_agg.merge(
                county_df_hz.rename(columns={"GEOID10": "GEOID", "NAME10": "County"}),
                on="GEOID", how="left",
            )

        choro_col, county_bar_col = st.columns([1.5, 1])

        with choro_col:
            if geo_json_hz is not None and not county_agg.empty:
                fig_choro = px.choropleth_mapbox(
                    county_agg,
                    geojson=geo_json_hz,
                    locations="GEOID",
                    featureidkey="properties.GEOID10",
                    color="value",
                    color_continuous_scale="Reds",
                    hover_name="County",
                    hover_data={"value": ":,.0f", "GEOID": False},
                    labels={"value": choro_label},
                    mapbox_style="carto-positron",
                    zoom=5.5,
                    center={"lat": 27.8, "lon": -81.5},
                    title=f"{hz_metric} by county",
                    height=480,
                )
                fig_choro.update_layout(
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    coloraxis_colorbar=dict(title=choro_label),
                )
                st.plotly_chart(fig_choro, use_container_width=True)
            else:
                st.info("County shapefile not available for choropleth map.")

        with county_bar_col:
            top_county_agg = (top_county_agg
                              .sort_values("value", ascending=False)
                              .head(20))
            fig_county_bar = px.bar(
                top_county_agg, x="value", y="CZ_NAME",
                orientation="h",
                labels={"value": choro_label, "CZ_NAME": "County / Zone"},
                title=f"Top 20 counties — {hz_metric}",
                color="value",
                color_continuous_scale="Blues",
            )
            fig_county_bar.update_layout(
                height=480, showlegend=False,
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_county_bar, use_container_width=True)

        # ── Hazard type × county pivot (shown when "All counties" selected) ──
        if hz_county == "All counties":
            st.markdown("**Hazard type breakdown per county (top 15 counties)**")
            top15 = (top_county_agg.head(15)["CZ_NAME"].tolist())
            pivot_df = dff[dff["CZ_NAME"].isin(top15)].copy()
            if metric_col is None:
                pivot = (pivot_df.groupby(["CZ_NAME", "HAZARD"])
                                 .size()
                                 .reset_index(name="value")
                                 .pivot(index="CZ_NAME", columns="HAZARD", values="value")
                                 .fillna(0).astype(int))
            else:
                pivot = (pivot_df.groupby(["CZ_NAME", "HAZARD"])[metric_col]
                                 .sum()
                                 .reset_index(name="value")
                                 .pivot(index="CZ_NAME", columns="HAZARD", values="value")
                                 .fillna(0).astype(int))
            pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
            pivot.columns.name = None
            pivot.index.name = "County / Zone"
            st.dataframe(pivot, use_container_width=True)

        # ── Number of hazards per county — map + table ───────────────────────
        st.markdown("---")
        st.markdown("**Number of hazard events per county**")

        county_event_count = (
            dff.groupby("CZ_NAME")
            .agg(
                Total_Events=("EVENT_TYPE", "count"),
                Property_Damage=("ADJ_DAMAGE_PROPERTY", "sum"),
                Deaths=("TOTAL_DEATHS", "sum"),
                Injuries=("TOTAL_INJURIES", "sum"),
            )
            .reset_index()
            .rename(columns={"CZ_NAME": "County / Zone"})
            .sort_values("Total_Events", ascending=False)
            .reset_index(drop=True)
        )

        # Map: event count choropleth (clickable)
        if geo_json_hz is not None and county_df_hz is not None:
            _cnt_geoid = (
                dff.groupby("GEOID").size().reset_index(name="Total_Events")
            )
            _cnt_geoid = _cnt_geoid.merge(
                county_df_hz.rename(columns={"GEOID10": "GEOID", "NAME10": "County"}),
                on="GEOID", how="left",
            )
            fig_cnt_map = px.choropleth_mapbox(
                _cnt_geoid,
                geojson=geo_json_hz,
                locations="GEOID",
                featureidkey="properties.GEOID10",
                color="Total_Events",
                color_continuous_scale="Oranges",
                hover_name="County",
                hover_data={"Total_Events": ":,", "GEOID": False},
                labels={"Total_Events": "# Events"},
                mapbox_style="carto-positron",
                zoom=5.5,
                center={"lat": 27.8, "lon": -81.5},
                title="Total hazard events by county — click a county for details",
                height=480,
            )
            fig_cnt_map.update_layout(
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                coloraxis_colorbar=dict(title="# Events"),
            )
            _cnt_map_event = st.plotly_chart(
                fig_cnt_map, use_container_width=True,
                on_select="rerun", selection_mode="points",
                key="hz_cnt_map",
            )

            # Handle county click
            _clicked_hz_geoid = None
            if (_cnt_map_event and _cnt_map_event.selection
                    and _cnt_map_event.selection.get("points")):
                _clicked_hz_geoid = _cnt_map_event.selection["points"][0].get("location")

            if _clicked_hz_geoid and county_df_hz is not None:
                _geoid_name_map = dict(zip(county_df_hz["GEOID10"], county_df_hz["NAME10"]))
                _clicked_hz_name = _geoid_name_map.get(_clicked_hz_geoid, "")
                if _clicked_hz_name:
                    dff_county = dff[dff["CZ_NAME"] == _clicked_hz_name]
                    st.markdown(f"### {_clicked_hz_name} County — hazard details")

                    # KPI row
                    _ck1, _ck2, _ck3, _ck4 = st.columns(4)
                    _ck1.metric("Total events",      f"{len(dff_county):,}")
                    _ck2.metric("Property damage",   f"${dff_county['ADJ_DAMAGE_PROPERTY'].sum()/1e6:.1f} M")
                    _ck3.metric("Deaths",            f"{int(dff_county['TOTAL_DEATHS'].sum()):,}")
                    _ck4.metric("Injuries",          f"{int(dff_county['TOTAL_INJURIES'].sum()):,}")

                    _det_col1, _det_col2 = st.columns(2)

                    # Top event types bar chart
                    with _det_col1:
                        _top_types = (
                            dff_county.groupby("EVENT_TYPE")
                            .size()
                            .reset_index(name="Count")
                            .sort_values("Count", ascending=False)
                            .head(10)
                        )
                        fig_det_bar = px.bar(
                            _top_types, x="Count", y="EVENT_TYPE",
                            orientation="h",
                            title=f"Top event types — {_clicked_hz_name}",
                            labels={"EVENT_TYPE": "Event type", "Count": "# Events"},
                            color="Count", color_continuous_scale="Oranges",
                        )
                        fig_det_bar.update_layout(
                            height=380, showlegend=False,
                            yaxis=dict(autorange="reversed"),
                            coloraxis_showscale=False,
                            margin={"t": 50, "b": 10},
                        )
                        st.plotly_chart(fig_det_bar, use_container_width=True)

                    # Events per year line chart
                    with _det_col2:
                        _ts_county = (
                            dff_county.groupby("start_year")
                            .size()
                            .reset_index(name="Count")
                        )
                        fig_det_ts = px.line(
                            _ts_county, x="start_year", y="Count",
                            title=f"Events per year — {_clicked_hz_name}",
                            labels={"start_year": "Year", "Count": "# Events"},
                            markers=True,
                        )
                        fig_det_ts.update_traces(line_color="#e8610a", line_width=2,
                                                  marker=dict(size=5))
                        fig_det_ts.update_layout(
                            height=380,
                            plot_bgcolor="#f8f9fa",
                            margin={"t": 50, "b": 10},
                        )
                        st.plotly_chart(fig_det_ts, use_container_width=True)

                    # Hazard type breakdown table
                    _hz_breakdown = (
                        dff_county.groupby("HAZARD")
                        .agg(Events=("EVENT_TYPE", "count"),
                             Damage=("ADJ_DAMAGE_PROPERTY", "sum"),
                             Deaths=("TOTAL_DEATHS", "sum"),
                             Injuries=("TOTAL_INJURIES", "sum"))
                        .sort_values("Events", ascending=False)
                        .reset_index()
                    )
                    _hz_breakdown["Damage"] = _hz_breakdown["Damage"].apply(
                        lambda x: f"${x/1e6:.1f} M" if x >= 1e6 else f"${x:,.0f}"
                    )
                    _hz_breakdown["Deaths"]   = _hz_breakdown["Deaths"].astype(int)
                    _hz_breakdown["Injuries"] = _hz_breakdown["Injuries"].astype(int)
                    st.dataframe(
                        _hz_breakdown.rename(columns={"HAZARD": "Hazard type",
                                                       "Damage": "Property Damage"}),
                        use_container_width=True, hide_index=True,
                    )
            else:
                st.caption("Click a county on the map to see its hazard details.")

        # Table
        county_event_count["Property_Damage"] = county_event_count["Property_Damage"].apply(
            lambda x: f"${x/1e6:.1f} M" if x >= 1e6 else f"${x:,.0f}"
        )
        county_event_count["Deaths"]   = county_event_count["Deaths"].astype(int)
        county_event_count["Injuries"] = county_event_count["Injuries"].astype(int)
        county_event_count.index = county_event_count.index + 1
        county_event_count.index.name = "Rank"

        st.dataframe(
            county_event_count.rename(columns={
                "Total_Events": "# Hazard Events",
                "Property_Damage": "Property Damage",
            }),
            use_container_width=True,
        )

        # ── Map — event locations ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Event locations**")
        map_df = dff.dropna(subset=["BEGIN_LAT", "BEGIN_LON"]).copy()
        map_df = map_df[
            map_df["BEGIN_LAT"].between(24.0, 31.5) &
            map_df["BEGIN_LON"].between(-87.7, -79.8)
        ]

        if map_df.empty:
            st.info("No events with coordinates for this selection.")
        else:
            _hz_bmap_opts = {
                "Streets (OpenStreetMap)": "open-street-map",
                "Light (Carto)":           "carto-positron",
                "Dark (Carto)":            "carto-darkmatter",
            }
            hz_bmap_style = st.selectbox(
                "Basemap", options=list(_hz_bmap_opts.keys()),
                index=1, key="hz_basemap", label_visibility="collapsed",
            )

            sample = map_df.sample(min(5000, len(map_df)), random_state=42)
            fig_hz_map = px.scatter_mapbox(
                sample,
                lat="BEGIN_LAT", lon="BEGIN_LON",
                color="HAZARD",
                hover_data={
                    "EVENT_TYPE": True,
                    "CZ_NAME": True,
                    "start_year": True,
                    "ADJ_DAMAGE_PROPERTY": ":,.0f",
                    "TOTAL_DEATHS": True,
                    "BEGIN_LAT": False,
                    "BEGIN_LON": False,
                },
                labels={
                    "CZ_NAME": "County/Zone",
                    "start_year": "Year",
                    "ADJ_DAMAGE_PROPERTY": "Damage ($)",
                    "TOTAL_DEATHS": "Deaths",
                },
                zoom=5.5,
                center={"lat": 27.8, "lon": -81.5},
                mapbox_style=_hz_bmap_opts[hz_bmap_style],
                title=f"Event locations (up to 5,000 shown) — {hz_metric}",
                height=520,
            )
            fig_hz_map.update_traces(marker=dict(size=5, opacity=0.7))
            fig_hz_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0},
                                     legend=dict(font=dict(size=10)))
            st.plotly_chart(fig_hz_map, use_container_width=True)

        # ── Download ──────────────────────────────────────────────────────────
        st.markdown("---")
        dl_cols = [
            "start_year", "EVENT_TYPE", "HAZARD", "CZ_NAME",
            "ADJ_DAMAGE_PROPERTY", "TOTAL_DEATHS", "TOTAL_INJURIES",
            "BEGIN_LAT", "BEGIN_LON",
        ]
        dl_hz = dff[dl_cols].rename(columns={
            "start_year": "Year", "CZ_NAME": "County_Zone",
            "ADJ_DAMAGE_PROPERTY": "Damage_USD_adj",
        })
        st.download_button(
            label="Download filtered hazards data (CSV)",
            data=dl_hz.to_csv(index=False).encode("utf-8"),
            file_name=f"florida_hazards_{hz_year_range[0]}_{hz_year_range[1]}.csv",
            mime="text/csv",
            use_container_width=False,
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Florida Population by Elevation (2010–2025)  |  "
    "Author: Bellah Harandi  |  "
    "Supervisors: Ivan David Haigh, Thomas Wahl, Christopher Emrich  |  "
    "University of Central Florida (UCF)  |  2026  |  "
    "Data: WorldPop 100 m rasters + USGS 1/3 arc-second DEM"
)
