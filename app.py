import re
import math
from io import BytesIO

import numpy as np
import pandas as pd
import folium
import pgeocode
import streamlit as st

from branca.colormap import linear
from streamlit_folium import st_folium


st.set_page_config(page_title="London Donor Map", layout="wide")

LONDON_CENTER = [51.5074, -0.1278]
LONDON_BOUNDS = {
    "min_lat": 51.28,
    "max_lat": 51.70,
    "min_lon": -0.53,
    "max_lon": 0.33,
}


def normalize_postcode(pc):
    if pd.isna(pc):
        return ""
    pc = str(pc).strip().upper()
    pc = re.sub(r"\s+", "", pc)
    if len(pc) > 3:
        pc = pc[:-3] + " " + pc[-3:]
    return pc


def detect_postcode_column(df):
    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["postcode", "post code", "postal", "zip"])
    ]
    if candidates:
        return candidates[0]

    postcode_pattern = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$", re.IGNORECASE)
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(50)
        matches = sum(bool(postcode_pattern.match(normalize_postcode(v))) for v in sample)
        if matches >= max(3, len(sample) // 4):
            return col

    raise ValueError("Could not detect a postcode column.")


def first_non_null(series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return s.iloc[0]


def jitter_point(lat, lon, idx, n, spread=0.0007):
    if n <= 1:
        return lat, lon

    angle = 2 * math.pi * (idx / n)
    radius = spread * (1 + idx / n)
    dlat = radius * math.cos(angle)
    dlon = radius * math.sin(angle) / max(math.cos(math.radians(lat)), 0.3)
    return lat + dlat, lon + dlon


def build_popup_html(row, gifts_col, gift_date_col, engagement_date_col):
    def fmt_date(v):
        return v.strftime("%Y-%m-%d") if pd.notna(v) else ""

    def fmt_num(v):
        return f"{v:,.0f}" if pd.notna(v) else ""

    fields = [
        ("Full Name", row["Full Name"]),
        ("Total Gifts", fmt_num(row[gifts_col])),
        ("Most Recent Gift Date", fmt_date(row[gift_date_col])),
        ("Most Recent Engagement Date", fmt_date(row[engagement_date_col])),
        ("Post Code", row.get("_postcode_clean", "")),
    ]

    rows = []
    for label, value in fields:
        if value != "" and pd.notna(value):
            rows.append(
                f"<tr><th style='text-align:left;padding-right:10px;vertical-align:top;'>{label}</th>"
                f"<td>{value}</td></tr>"
            )

    return f"""
    <div style="font-size:13px; line-height:1.4;">
      <table style="border-collapse:collapse;">
        {''.join(rows)}
      </table>
    </div>
    """


@st.cache_resource
def get_nomi():
    return pgeocode.Nominatim("gb")


def load_and_prepare(file_bytes, postcode_col, gifts_col, gift_date_col, engagement_date_col):
    df = pd.read_csv(BytesIO(file_bytes), encoding="cp1252")
    df.columns = df.columns.str.strip()

    required = ["First Name", "Last Name", postcode_col, gifts_col, gift_date_col, engagement_date_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}")

    df["First Name"] = df["First Name"].fillna("").astype(str).str.strip()
    df["Last Name"] = df["Last Name"].fillna("").astype(str).str.strip()
    df["Full Name"] = (df["First Name"] + " " + df["Last Name"]).str.strip()

    df[gifts_col] = pd.to_numeric(df[gifts_col], errors="coerce")
    df[gift_date_col] = pd.to_datetime(df[gift_date_col], errors="coerce")
    df[engagement_date_col] = pd.to_datetime(df[engagement_date_col], errors="coerce")

    donors = (
        df.groupby("Full Name", as_index=False)
          .agg({
              "First Name": first_non_null,
              "Last Name": first_non_null,
              postcode_col: first_non_null,
              gifts_col: "sum",
              gift_date_col: "max",
              engagement_date_col: "max",
          })
    )

    donors["_postcode_clean"] = donors[postcode_col].apply(normalize_postcode)

    nomi = get_nomi()
    geo = nomi.query_postal_code(donors["_postcode_clean"].tolist())
    donors["latitude"] = geo["latitude"].values
    donors["longitude"] = geo["longitude"].values

    donors = donors.dropna(subset=["latitude", "longitude"]).copy()

    donors = donors[
        (donors["latitude"] >= LONDON_BOUNDS["min_lat"]) &
        (donors["latitude"] <= LONDON_BOUNDS["max_lat"]) &
        (donors["longitude"] >= LONDON_BOUNDS["min_lon"]) &
        (donors["longitude"] <= LONDON_BOUNDS["max_lon"])
    ].copy()

    donors = donors.reset_index(drop=True)
    donors["_donor_id"] = [f"donor_{i}" for i in range(len(donors))]

    donors["_postcode_rank"] = donors.groupby("_postcode_clean").cumcount()
    donors["_postcode_group_size"] = donors.groupby("_postcode_clean")["_postcode_clean"].transform("size")

    jittered = donors.apply(
        lambda r: pd.Series(
            jitter_point(
                r["latitude"],
                r["longitude"],
                int(r["_postcode_rank"]),
                int(r["_postcode_group_size"])
            )
        ),
        axis=1
    )
    donors["plot_latitude"] = jittered[0]
    donors["plot_longitude"] = jittered[1]

    return donors


def build_map(donors, gifts_col, gift_date_col, engagement_date_col, selected_donor_ids=None):
    selected_donor_ids = set(selected_donor_ids or [])

    m = folium.Map(location=LONDON_CENTER, zoom_start=11, tiles="CartoDB positron")

    values = np.log1p(donors[gifts_col].fillna(0)) if len(donors) else pd.Series([0, 1])
    min_g = float(values.min())
    max_g = float(values.max())
    if min_g == max_g:
        max_g = min_g + 1

    colormap = linear.YlOrRd_09.scale(min_g, max_g)
    colormap.caption = "Total Gifts (log scale)"
    colormap.add_to(m)

    for _, row in donors.iterrows():
        gifts_value = 0 if pd.isna(row[gifts_col]) else float(row[gifts_col])
        colour_value = np.log1p(gifts_value)
        color = colormap(colour_value)

        selected = row["_donor_id"] in selected_donor_ids
        marker_color = "#111111" if selected else color
        marker_radius = 11 if selected else 6
        marker_weight = 4 if selected else 1
        fill_opacity = 1 if selected else 0.85

        folium.CircleMarker(
            location=[row["plot_latitude"], row["plot_longitude"]],
            radius=marker_radius,
            weight=marker_weight,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=fill_opacity,
            popup=folium.Popup(
                build_popup_html(row, gifts_col, gift_date_col, engagement_date_col),
                max_width=360,
            ),
            tooltip=row["Full Name"],
        ).add_to(m)

    if len(donors):
        sw = [donors["plot_latitude"].min(), donors["plot_longitude"].min()]
        ne = [donors["plot_latitude"].max(), donors["plot_longitude"].max()]
        m.fit_bounds([sw, ne])

    return m


def ranked_table(donors, sort_col, display_col_name, ascending=False):
    ranked = (
        donors.dropna(subset=[sort_col])
              .sort_values(sort_col, ascending=ascending)
              .head(10)
              .reset_index(drop=True)
    )
    view = ranked[["Full Name", sort_col]].copy()
    view.columns = ["Full Name", display_col_name]
    return ranked, view


def update_selected_ids(event, source_df, selected_ids):
    if hasattr(event, "selection") and event.selection.rows:
        for row_idx in event.selection.rows:
            selected_ids.add(source_df.iloc[row_idx]["_donor_id"])
    return selected_ids


st.title("London donor map")

uploaded = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded is None:
    st.info("Upload the CSV to begin.")
    st.stop()

file_bytes = uploaded.getvalue()

with st.sidebar:
    st.header("Columns")

    preview_df = pd.read_csv(BytesIO(file_bytes), encoding="cp1252")
    preview_df.columns = preview_df.columns.str.strip()

    default_postcode = detect_postcode_column(preview_df)
    postcode_col = st.text_input("Postcode column", value=default_postcode)
    gifts_col = st.text_input("Gift total column", value="Total Gifts - GBP")
    gift_date_col = st.text_input("Last gift date column", value="Last Gift Date")
    engagement_date_col = st.text_input("Last engagement date column", value="Last Engagement Date")

    if st.button("Clear all selections"):
        st.session_state.selected_ids_top_gifts = set()
        st.session_state.selected_ids_top_recent_gift = set()
        st.session_state.selected_ids_top_recent_engagement = set()

try:
    donors = load_and_prepare(
        file_bytes,
        postcode_col,
        gifts_col,
        gift_date_col,
        engagement_date_col,
    )
except Exception as e:
    st.error(f"Could not load the file: {e}")
    st.stop()

if donors.empty:
    st.warning("No valid London postcodes were found after cleaning and filtering.")
    st.stop()

top_gifts_ranked, top_gifts_view = ranked_table(donors, gifts_col, gifts_col, ascending=False)
top_recent_gift_ranked, top_recent_gift_view = ranked_table(donors, gift_date_col, gift_date_col, ascending=False)
top_recent_engagement_ranked, top_recent_engagement_view = ranked_table(donors, engagement_date_col, engagement_date_col, ascending=False)

if "selected_ids_top_gifts" not in st.session_state:
    st.session_state.selected_ids_top_gifts = set()

if "selected_ids_top_recent_gift" not in st.session_state:
    st.session_state.selected_ids_top_recent_gift = set()

if "selected_ids_top_recent_engagement" not in st.session_state:
    st.session_state.selected_ids_top_recent_engagement = set()

st.caption("Click one or more rows in any tab to highlight those donors on the map.")

map_col, table_col = st.columns([1.55, 1])

with table_col:
    st.subheader("Top 10 rankings")

    tab1, tab2, tab3 = st.tabs([
        "Highest total gifts",
        "Most recent gift date",
        "Most recent engagement date",
    ])

    with tab1:
        gifts_event = st.dataframe(
            top_gifts_view,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="top_gifts_table",
        )
        st.session_state.selected_ids_top_gifts = update_selected_ids(
            gifts_event,
            top_gifts_ranked,
            st.session_state.selected_ids_top_gifts,
        )

    with tab2:
        recent_gift_event = st.dataframe(
            top_recent_gift_view,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="top_recent_gift_table",
        )
        st.session_state.selected_ids_top_recent_gift = update_selected_ids(
            recent_gift_event,
            top_recent_gift_ranked,
            st.session_state.selected_ids_top_recent_gift,
        )

    with tab3:
        recent_engagement_event = st.dataframe(
            top_recent_engagement_view,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="top_recent_engagement_table",
        )
        st.session_state.selected_ids_top_recent_engagement = update_selected_ids(
            recent_engagement_event,
            top_recent_engagement_ranked,
            st.session_state.selected_ids_top_recent_engagement,
        )

    selected_ids = (
        set(st.session_state.selected_ids_top_gifts)
        | set(st.session_state.selected_ids_top_recent_gift)
        | set(st.session_state.selected_ids_top_recent_engagement)
    )

    selected_names = donors.loc[donors["_donor_id"].isin(selected_ids), "Full Name"].tolist()
    if selected_names:
        st.markdown("**Selected donors:** " + ", ".join(selected_names))
    else:
        st.markdown("**Selected donors:** none")

with map_col:
    st.subheader("Map")
    m = build_map(
        donors=donors,
        gifts_col=gifts_col,
        gift_date_col=gift_date_col,
        engagement_date_col=engagement_date_col,
        selected_donor_ids=selected_ids,
    )
    st_folium(m, width=1100, height=720)
