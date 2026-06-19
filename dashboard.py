"""
Florida Population by Elevation — Streamlit Dashboard
Author: Bella Harandi
Date: 2026

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
from PIL import Image
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
DEM_PATH   = os.path.join(_BASE, "data", "dem_florida_100m.tif")

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


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_parquet(DATA_PATH)


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


@st.cache_data(show_spinner="Reading DEM …")
def get_dem_overlay(geom_wkt: str, unit_k: str):
    """
    Clip the 10 m DEM to a county geometry, colorize with 5 elevation classes,
    and return (data_uri_png, [west, south, east, north], hover_dict).
    Returns (None, None, None) if the DEM is missing or clipping fails.
    """
    if not os.path.exists(DEM_PATH):
        return None, None, None

    from shapely import wkt as shapely_wkt
    geom_wgs84 = shapely_wkt.loads(geom_wkt)
    gdf = gpd.GeoDataFrame(geometry=[geom_wgs84], crs="EPSG:4326").to_crs("EPSG:4269")
    geom_4269 = gdf.geometry.iloc[0]

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

    from shapely import wkt as shapely_wkt
    geom_wgs84 = shapely_wkt.loads(geom_wkt)
    gdf = gpd.GeoDataFrame(geometry=[geom_wgs84], crs="EPSG:4326").to_crs("EPSG:4269")
    geom_4269 = gdf.geometry.iloc[0]

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

    MAX_PX = 600
    step_h = max(1, h // MAX_PX)
    step_w = max(1, w // MAX_PX)
    dem_ds          = dem[::step_h, ::step_w]
    poly_outside_ds = poly_outside[::step_h, ::step_w]
    valid           = ~np.isnan(dem_ds) & ~poly_outside_ds

    rgba = np.zeros((dem_ds.shape[0], dem_ds.shape[1], 4), dtype=np.uint8)
    rgba[valid & (dem_ds < 0)]                          = [33,  102, 172, 210]  # deep blue — already below sea level
    rgba[valid & (dem_ds >= 0) & (dem_ds <= sea_level_m)] = [214,  69,  65, 210]  # red — flooded
    rgba[poly_outside_ds]                               = [0,   0,   0,   0]   # transparent outside

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return data_uri, [west, south, east, north]


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
st.caption("Author: Bella Harandi")
st.caption("University of Central Florida (UCF)  |  2026")

if df_all is None:
    st.error(
        f"Data file not found: `{DATA_PATH}`\n\n"
        "Run **`create_sample_data.py`** or **`processing.ipynb`** first."
    )
    st.stop()

fl_geojson, county_meta = load_county_geojson()
state_rings = load_state_boundary()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

unit      = st.sidebar.radio("Elevation unit", ["Feet (ft)", "Metric (m)"], horizontal=True)
use_feet  = unit == "Feet (ft)"   # default: Feet
unit_key  = "Feet" if use_feet else "Metric"
band_order  = BAND_ORDER_FT  if use_feet else BAND_ORDER_M
band_colors = BAND_COLORS_FT if use_feet else BAND_COLORS_M
unit_label  = "ft above MSL" if use_feet else "m above MSL"

all_years = sorted(df_all["Year"].unique())

county_options = ["Florida (Statewide)"] + sorted(
    df_all[df_all["Scope"] == "County"]["County_Name"].unique()
)
selected_area = st.sidebar.selectbox("County / Statewide", county_options,
                                      key="county_selector")

# When unit changes, reset bands to the new unit's full list on the same rerun
# (setting the key directly avoids a 2-rerun flash that del would cause)
if "elev_bands" in st.session_state:
    stale = [b for b in st.session_state["elev_bands"] if b not in band_order]
    if stale:
        st.session_state["elev_bands"] = band_order

selected_bands = st.sidebar.multiselect(
    "Elevation bands", options=band_order, default=band_order, key="elev_bands",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Population source: WorldPop 100 m rasters  \n"
    "Elevation source: USGS 1/3 arc-second DEM  \n"
    "Geography: Florida (FIPS 12)  \n"
    "Author: Bella Harandi  \n"
    "University of Central Florida (UCF)  |  2026"
)


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


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Distribution", "Map", "Sea Level Rise"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Distribution (single year snapshot)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_ctrl, _ = st.columns([1, 3])
    snap_year   = col_ctrl.selectbox("Select year", all_years,
                                      index=len(all_years) - 1, key="snap_year")
    df_snap = df_area[df_area["Year"] == snap_year].sort_values("Elev_Band")

    if not selected_bands:
        st.info("Select at least one elevation band in the sidebar.")
    elif df_snap.empty:
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

            band_options = ["All elevations"] + band_order
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
                (df_all["Elev_Band"] == to_query_band(map_band, use_feet))
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
                            if clicked_name in map_county_options:
                                st.session_state["map_county_sel"] = clicked_name
                                st.rerun()

        # ── Detail table — full width below map ───────────────────────────────
        detail_label = map_county if map_county != "All counties" else "Florida (Statewide)"
        det_scope = "Statewide" if map_county == "All counties" else "County"
        detail = df_all[
            (df_all["Scope"] == det_scope) &
            (df_all["Year"]  == map_year)
        ].copy()
        detail = to_display_bands(detail, use_feet)
        if map_county != "All counties":
            detail = detail[detail["County_Name"] == map_county]

        detail["Elev_Band"] = pd.Categorical(
            detail["Elev_Band"], categories=band_order, ordered=True)
        detail = detail.sort_values("Elev_Band")

        band_col = f"Band ({unit_label})"
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
        if map_county != "All counties" and not df_map.empty:
            st.markdown("---")
            zoom_col1, zoom_col2 = st.columns(2)

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
                center_lat, center_lon, zoom_level = 27.5, -81.5, 7

            # ── Zoom map with DEM overlay ─────────────────────────────────────
            with zoom_col1:
                st.markdown(f"**{map_county} — elevation map (DEM)**")

                dem_img, dem_bounds, dem_hover = get_dem_overlay(geom.wkt, unit_key)

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
                if geom.geom_type == "MultiPolygon":
                    boundary_lons, boundary_lats = [], []
                    for poly in geom.geoms:
                        coords = list(poly.exterior.coords)
                        boundary_lons += [c[0] for c in coords] + [None]
                        boundary_lats += [c[1] for c in coords] + [None]
                else:
                    coords = list(geom.exterior.coords)
                    boundary_lons = [c[0] for c in coords]
                    boundary_lats = [c[1] for c in coords]

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
                st.plotly_chart(fig_zoom, use_container_width=True)

                if dem_img is not None and show_dem:
                    st.markdown(_dem_legend_html(unit_key), unsafe_allow_html=True)
                elif dem_img is None:
                    st.warning("DEM file not found — outline only.")

            # ── Elevation profile chart ───────────────────────────────────────
            with zoom_col2:
                st.markdown(f"**{map_county} — elevation profile ({map_year})**")

                elev_profile = df_all[
                    (df_all["Scope"]       == "County") &
                    (df_all["Year"]        == map_year) &
                    (df_all["County_Name"] == map_county)
                ].copy()
                elev_profile = to_display_bands(elev_profile, use_feet)
                elev_profile["Elev_Band"] = pd.Categorical(
                    elev_profile["Elev_Band"], categories=band_order, ordered=True)
                elev_profile = elev_profile.sort_values("Elev_Band")

                # Build a silhouette-style area chart using Elev_Min as x
                fig_profile = go.Figure()
                for _, row in elev_profile.iterrows():
                    color = band_colors.get(row["Elev_Band"], "#888888")
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
                    xaxis_title=f"Elevation ({unit_label})",
                    yaxis_title="Population",
                    showlegend=False,
                    height=400,
                    margin={"r": 10, "t": 50, "l": 10, "b": 50},
                    plot_bgcolor="#f8f9fa",
                    xaxis=dict(categoryorder="array",
                                categoryarray=band_order),
                )
                st.plotly_chart(fig_profile, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # STATEWIDE DEM — shown when no county is selected
        # ══════════════════════════════════════════════════════════════════════
        elif map_county == "All counties":
            st.markdown("---")
            state_col1, state_col2 = st.columns(2)

            # ── Statewide DEM map ─────────────────────────────────────────────
            with state_col1:
                st.markdown("**Florida — elevation map (DEM)**")
                state_wkt = load_state_geometry_wkt()
                if state_wkt:
                    dem_img, dem_bounds, dem_hover = get_dem_overlay(state_wkt, unit_key)

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
                    st.plotly_chart(fig_state, use_container_width=True)

                    if dem_img is not None and show_state_dem:
                        st.markdown(_dem_legend_html(unit_key), unsafe_allow_html=True)
                    elif dem_img is None:
                        st.warning("DEM file not found — outline only.")

            # ── Statewide elevation profile chart ────────────────────────────
            with state_col2:
                st.markdown(f"**Florida — elevation profile ({map_year})**")

                elev_profile_state = df_all[
                    (df_all["Scope"] == "Statewide") &
                    (df_all["Year"]  == map_year)
                ].copy()
                elev_profile_state = to_display_bands(elev_profile_state, use_feet)
                elev_profile_state["Elev_Band"] = pd.Categorical(
                    elev_profile_state["Elev_Band"], categories=band_order, ordered=True)
                elev_profile_state = elev_profile_state.sort_values("Elev_Band")

                fig_state_profile = go.Figure()
                for _, row in elev_profile_state.iterrows():
                    color = band_colors.get(row["Elev_Band"], "#888888")
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
                    xaxis_title=f"Elevation ({unit_label})",
                    yaxis_title="Population",
                    showlegend=False,
                    height=460,
                    margin={"r": 10, "t": 50, "l": 10, "b": 50},
                    plot_bgcolor="#f8f9fa",
                    xaxis=dict(categoryorder="array", categoryarray=band_order),
                )
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
    st.caption("Areas shown in red would be below the tideline at the selected sea level rise scenario.")

    slr_col1, slr_col2 = st.columns([3, 1])

    with slr_col2:
        slr_area = st.selectbox(
            "County / Statewide", county_options, key="slr_area",
        )
        slr_year = st.selectbox(
            "Year", all_years, index=len(all_years) - 1, key="slr_year",
        )
        if use_feet:
            slr_ft   = st.slider("Sea level rise (ft)", 0.0, 30.0, 1.0, 0.5, key="slr_slider")
            slr_m    = slr_ft / 3.28084
            slr_label = f"{slr_ft:.1f} ft"
        else:
            slr_m    = st.slider("Sea level rise (m)", 0.0, 10.0, 0.3, 0.1, key="slr_slider")
            slr_label = f"{slr_m:.1f} m"

        _slr_basemap_map = {
            "Streets (OpenStreetMap)": "open-street-map",
            "Light (Carto)":           "carto-positron",
            "Dark (Carto)":            "carto-darkmatter",
        }
        slr_basemap_style = st.selectbox(
            "Basemap", options=list(_slr_basemap_map.keys()), index=1, key="slr_basemap",
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
                    if f["properties"]["GEOID10"] == slr_geoid] if slr_geoid else []
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
            elif flood_img is None and os.path.exists(DEM_PATH):
                st.info("Zoom in or select a smaller area if the map is slow to load.")

            fig_slr.update_layout(
                mapbox=mapbox_cfg_slr,
                height=520,
                margin={"r": 0, "t": 10, "l": 0, "b": 0},
                uirevision=f"{slr_area}_{slr_m}",
            )
            st.plotly_chart(fig_slr, use_container_width=True)

            # Legend
            st.markdown(
                '<span style="display:inline-block;width:14px;height:14px;background:#d64541;'
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

    at_risk_display = to_display_bands(at_risk_df.copy(), use_feet)
    at_risk_display["Elev_Band"] = pd.Categorical(
        at_risk_display["Elev_Band"], categories=band_order, ordered=True)
    at_risk_display = at_risk_display.sort_values("Elev_Band")
    at_risk_display["Status"] = at_risk_display["at_risk"].map(
        {True: "At risk", False: "Safe"})
    st.dataframe(
        at_risk_display[["Elev_Band", "Population", "Pct_of_State", "Status"]]
        .rename(columns={"Elev_Band": f"Elevation ({unit_label})", "Pct_of_State": "% State"})
        .reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Florida Population by Elevation (2010–2025)  |  "
    "Author: Bella Harandi  |  University of Central Florida  |  2026  |  "
    "Data: WorldPop 100 m rasters + USGS 1/3 arc-second DEM"
)
