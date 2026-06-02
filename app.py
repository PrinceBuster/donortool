import re
import math

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


def load_and_prepare(uploaded_file, postcode_col, gifts_col, gift_date_col, engagement_date_col):
    df = pd.read_csv(uploaded_file, encoding="cp1252")
    df.columns = df.columns.str.strip()

    for col in ["First Name", "Last Name", postcode_col, gifts_col, gift_date_col, engagement_date_col]:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

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


def top10_table(donors, sort_col, ascending=False):
    out = donors.dropna(subset=[sort_col]).sort_values(sort_col, ascending=ascending).head(10).copy()
    out = out[["Full Name"]].reset_index(drop=True)
    out.index = np.arange(1, len(out) + 1)
    out.index.name = "Rank"
    return out


def build_map(donors, gifts_col, gift_date_col, engagement_date_col, selected_donor_id=None):
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

        selected = selected_donor_id == row["_donor_id"]
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


st.title("London donor map")

uploaded = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded is None:
    st.info("Upload the CSV to begin.")
    st.stop()

with st.sidebar:
    st.header("Columns")

    df_preview = pd.read_csv(uploaded, encoding="cp1252")
    df_preview.columns = df_preview.columns.str.strip()

    default_postcode = detect_postcode_column(df_preview)
    postcode_col = st.text_input("Postcode column", value=default_postcode)
    gifts_col = st.text_input("Gift total column", value="Total Gifts - GBP")
    gift_date_col = st.text_input("Last gift date column", value="Last Gift Date")
    engagement_date_col = st.text_input("Last engagement date column", value="Last Engagement Date")

    highlight_mode = st.radio(
        "Highlight donor from",
        ["All donors", "Top gifts", "Top recent gift", "Top recent engagement"],
        index=0,
    )

try:
    donors = load_and_prepare(uploaded, postcode_col, gifts_col, gift_date_col, engagement_date_col)
except Exception as e:
    st.error(f"Could not load the file: {e}")
    st.stop()

if donors.empty:
    st.warning("No valid London postcodes were found after cleaning and filtering.")
    st.stop()

top_gifts_df = top10_table(donors, gifts_col, ascending=False)
top_recent_gift_df = top10_table(donors, gift_date_col, ascending=False)
top_recent_engagement_df = top10_table(donors, engagement_date_col, ascending=False)

if "selected_name" not in st.session_state:
    st.session_state.selected_name = donors["Full Name"].iloc[0]

all_names = donors["Full Name"].tolist()

if highlight_mode == "All donors":
    name_options = all_names
elif highlight_mode == "Top gifts":
    name_options = top_gifts_df["Full Name"].tolist()
elif highlight_mode == "Top recent gift":
    name_options = top_recent_gift_df["Full Name"].tolist()
else:
    name_options = top_recent_engagement_df["Full Name"].tolist()

if st.session_state.selected_name not in name_options:
    st.session_state.selected_name = name_options[0]

selected_name = st.sidebar.selectbox(
    "Choose donor to highlight",
    options=name_options,
    index=name_options.index(st.session_state.selected_name),
)

st.session_state.selected_name = selected_name
selected_donor_id = donors.loc[donors["Full Name"] == selected_name, "_donor_id"].iloc[0]

left, right = st.columns([1.5, 1])

with left:
    st.subheader("Map")
    m = build_map(
        donors=donors,
        gifts_col=gifts_col,
        gift_date_col=gift_date_col,
        engagement_date_col=engagement_date_col,
        selected_donor_id=selected_donor_id,
    )
    st_folium(m, width=1100, height=720)

with right:
    st.subheader("Top 10 rankings")

    tab1, tab2, tab3 = st.tabs([
        "Highest total gifts",
        "Most recent gift date",
        "Most recent engagement date",
    ])

    with tab1:
        st.dataframe(top_gifts_df, use_container_width=True, height=360)
        st.caption("Use the sidebar dropdown to highlight a donor from this list.")

    with tab2:
        st.dataframe(top_recent_gift_df, use_container_width=True, height=360)
        st.caption("Use the sidebar dropdown to highlight a donor from this list.")

    with tab3:
        st.dataframe(top_recent_engagement_df, use_container_width=True, height=360)
        st.caption("Use the sidebar dropdown to highlight a donor from this list.")
