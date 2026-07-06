import warnings
warnings.filterwarnings("ignore")

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# 1. PAGE SETTING
# =========================================================

st.set_page_config(
    page_title="Seoul Population Intelligence Dashboard",
    page_icon="🌏",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #070B14 0%, #0F172A 45%, #111827 100%);
        color: #F8FAFC;
    }
    h1, h2, h3 {
        color: #F8FAFC;
        font-weight: 800;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .stMetric {
        background-color: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 16px;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricValue"] {
        color: #38BDF8;
    }
    div[data-testid="stMetricDelta"] {
        color: #22C55E;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 2. BASIC CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_NATIVE_FILE = DATA_DIR / "SEOUL_STYTIME_04_250M_OPEN_NATIVE_20260629.csv"
DEFAULT_FOREIGN_FILE = DATA_DIR / "SEOUL_STYTIME_05_250M_OPEN_FORN_LONG_20260629.csv"


# =========================================================
# 3. UTILITY FUNCTIONS
# =========================================================

def read_csv_auto(source):
    """Read CSV with common Korean encodings. Works with local path or Streamlit uploaded file."""
    encodings = ["cp949", "utf-8-sig", "utf-8"]

    for enc in encodings:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue

    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source, low_memory=False)


def clean_columns(df):
    df = df.copy()
    df.columns = [
        str(col)
        .replace("\ufeff", "")
        .replace("?", "")
        .replace('"', "")
        .replace("'", "")
        .strip()
        for col in df.columns
    ]
    return df


def to_numeric_safe(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("*", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def find_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    for col in df.columns:
        col_text = str(col).replace(" ", "")
        for candidate in candidates:
            candidate_text = str(candidate).replace(" ", "")
            if candidate_text in col_text:
                return col

    return None


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def style_plotly(fig):
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="#F8FAFC",
        title_font_color="#F8FAFC",
        legend_font_color="#F8FAFC",
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig


def detect_age_label(col):
    text = str(col)

    if "인구" not in text and "생활인구" not in text:
        return None

    blocked_words = [
        "총합", "총인구", "인구수총합", "인구수 총합",
        "내국인", "외국인", "총체류", "Total", "Native", "Foreign"
    ]

    if any(word in text for word in blocked_words):
        return None

    match = re.search(r"(\d{1,2})\s*세.*?(\d{1,2})\s*세", text)
    if match:
        return f"{int(match.group(1))}-{int(match.group(2))}"

    match = re.search(r"(\d{1,2})\s*대", text)
    if match:
        return f"{int(match.group(1))}s"

    match = re.search(r"(\d{1,2})\s*세\s*이상", text)
    if match:
        return f"{int(match.group(1))}+"

    return None


def age_sort_key(label):
    match = re.search(r"\d+", str(label))
    if match:
        return int(match.group())
    return 999


def try_add_lat_lon(df):
    df = df.copy()

    if "X" not in df.columns or "Y" not in df.columns:
        return df, False, "No X/Y columns"

    x = pd.to_numeric(df["X"], errors="coerce")
    y = pd.to_numeric(df["Y"], errors="coerce")

    # Already longitude / latitude
    if x.between(126, 128).mean() > 0.8 and y.between(37, 38).mean() > 0.8:
        df["lon"] = x
        df["lat"] = y
        return df, True, "Original coordinates are longitude/latitude"

    try:
        from pyproj import Transformer
    except Exception:
        return df, False, "pyproj is not installed. X/Y map will be used."

    candidate_epsg = [5186, 5179, 5181, 5174, 5178, 3857]
    sample = df[["X", "Y"]].dropna()

    if len(sample) == 0:
        return df, False, "No valid coordinate rows"

    sample = sample.sample(min(5000, len(sample)), random_state=42)

    best_epsg = None
    best_score = -1

    for epsg in candidate_epsg:
        try:
            transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
            lon_sample, lat_sample = transformer.transform(sample["X"].values, sample["Y"].values)
            lon_sample = pd.Series(lon_sample)
            lat_sample = pd.Series(lat_sample)

            score = (
                lon_sample.between(126.6, 127.4).mean()
                + lat_sample.between(37.2, 37.9).mean()
            )

            if score > best_score:
                best_score = score
                best_epsg = epsg
        except Exception:
            pass

    if best_epsg is not None and best_score >= 1.2:
        transformer = Transformer.from_crs(f"EPSG:{best_epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(df["X"].values, df["Y"].values)
        df["lon"] = lon
        df["lat"] = lat

        valid_ratio = (df["lon"].between(126.6, 127.4) & df["lat"].between(37.2, 37.9)).mean()
        if valid_ratio > 0.5:
            return df, True, f"Converted from EPSG:{best_epsg}"

    return df, False, "Coordinate conversion failed. X/Y map will be used."


# =========================================================
# 4. DATA LOADING
# =========================================================

def get_data_sources():
    local_native_exists = DEFAULT_NATIVE_FILE.exists()
    local_foreign_exists = DEFAULT_FOREIGN_FILE.exists()

    st.sidebar.header("Data Source")

    if local_native_exists and local_foreign_exists:
        st.sidebar.success("Local data files detected in /data folder.")
        return DEFAULT_NATIVE_FILE, DEFAULT_FOREIGN_FILE

    st.sidebar.warning("Upload CSV files or place them inside the /data folder.")

    native_upload = st.sidebar.file_uploader(
        "Upload native population CSV",
        type=["csv"],
        key="native_upload"
    )

    foreign_upload = st.sidebar.file_uploader(
        "Upload foreign population CSV",
        type=["csv"],
        key="foreign_upload"
    )

    if native_upload is None or foreign_upload is None:
        st.title("Seoul Population Intelligence Dashboard")
        st.info(
            "To start, upload both CSV files from the sidebar. "
            "For GitHub use, create a data folder and put the two CSV files inside it."
        )
        st.code(
            """
project-folder/
├── world.py
├── requirements.txt
└── data/
    ├── SEOUL_STYTIME_04_250M_OPEN_NATIVE_20260629.csv
    └── SEOUL_STYTIME_05_250M_OPEN_FORN_LONG_20260629.csv
            """.strip()
        )
        st.stop()

    return native_upload, foreign_upload


@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_data(native_source, foreign_source):
    native = read_csv_auto(native_source)
    foreign = read_csv_auto(foreign_source)

    native = clean_columns(native)
    foreign = clean_columns(foreign)

    native_grid_col = find_column(native, ["격자 ID", "격자ID", "Grid ID", "grid_id"])
    native_x_col = find_column(native, ["250m 격자X좌표", "격자X좌표", "X좌표", "X"])
    native_y_col = find_column(native, ["250m 격자Y좌표", "격자Y좌표", "Y좌표", "Y"])
    native_hour_col = find_column(native, ["체류 시작 시간", "체류시작시간", "시간대", "Hour"])
    native_duration_col = find_column(native, ["체류 시간 코드(분)", "체류시간", "체류 시간", "Stay Duration"])
    native_pop_col = find_column(native, ["인구수 총합", "총합", "총인구", "Native Population"])
    native_dong_col = find_column(native, ["행정동 코드", "행정동코드", "Administrative Dong Code"])

    foreign_grid_col = find_column(foreign, ["격자 ID", "격자ID", "Grid ID", "grid_id"])
    foreign_x_col = find_column(foreign, ["250m 격자X좌표", "격자X좌표", "X좌표", "X"])
    foreign_y_col = find_column(foreign, ["250m 격자Y좌표", "격자Y좌표", "Y좌표", "Y"])
    foreign_hour_col = find_column(foreign, ["체류 시작 시간", "체류시작시간", "시간대", "Hour"])
    foreign_duration_col = find_column(foreign, ["체류 시간 코드(분)", "체류시간", "체류 시간", "Stay Duration"])
    foreign_pop_col = find_column(foreign, ["인구수", "외국인인구", "Foreign Population"])
    foreign_nation_col = find_column(foreign, ["국적명", "국적", "국가명", "Nationality", "Country"])
    foreign_dong_col = find_column(foreign, ["행정동 코드", "행정동코드", "Administrative Dong Code"])

    missing_native = []
    for label, col in {
        "Grid ID": native_grid_col,
        "X": native_x_col,
        "Y": native_y_col,
        "Hour": native_hour_col,
        "Stay Duration": native_duration_col,
        "Native Population": native_pop_col,
    }.items():
        if col is None:
            missing_native.append(label)

    missing_foreign = []
    for label, col in {
        "Grid ID": foreign_grid_col,
        "Hour": foreign_hour_col,
        "Stay Duration": foreign_duration_col,
        "Foreign Population": foreign_pop_col,
    }.items():
        if col is None:
            missing_foreign.append(label)

    if missing_native:
        st.error(f"Missing columns in native population file: {missing_native}")
        st.write(native.columns.tolist())
        st.stop()

    if missing_foreign:
        st.error(f"Missing columns in foreign population file: {missing_foreign}")
        st.write(foreign.columns.tolist())
        st.stop()

    native_rename = {
        native_grid_col: "Grid ID",
        native_x_col: "X",
        native_y_col: "Y",
        native_hour_col: "Hour",
        native_duration_col: "Stay Duration",
        native_pop_col: "Native Population",
    }

    if native_dong_col is not None:
        native_rename[native_dong_col] = "Administrative Dong Code"

    foreign_rename = {
        foreign_grid_col: "Grid ID",
        foreign_hour_col: "Hour",
        foreign_duration_col: "Stay Duration",
        foreign_pop_col: "Foreign Population",
    }

    if foreign_x_col is not None:
        foreign_rename[foreign_x_col] = "X"
    if foreign_y_col is not None:
        foreign_rename[foreign_y_col] = "Y"
    if foreign_nation_col is not None:
        foreign_rename[foreign_nation_col] = "Nationality"
    if foreign_dong_col is not None:
        foreign_rename[foreign_dong_col] = "Administrative Dong Code"

    native = native.rename(columns=native_rename)
    foreign = foreign.rename(columns=foreign_rename)

    if "Administrative Dong Code" not in native.columns:
        native["Administrative Dong Code"] = "unknown"

    if "Nationality" not in foreign.columns:
        foreign["Nationality"] = "Unknown"

    for col in native.columns:
        if "인구" in str(col) or "생활인구" in str(col):
            native[col] = to_numeric_safe(native[col])

    for col in foreign.columns:
        if "인구" in str(col) or "생활인구" in str(col):
            foreign[col] = to_numeric_safe(foreign[col])

    for col in ["X", "Y", "Hour", "Stay Duration", "Native Population"]:
        native[col] = to_numeric_safe(native[col])

    for col in ["Hour", "Stay Duration", "Foreign Population"]:
        foreign[col] = to_numeric_safe(foreign[col])

    native["Grid ID"] = native["Grid ID"].astype(str)
    foreign["Grid ID"] = foreign["Grid ID"].astype(str)
    native["Administrative Dong Code"] = native["Administrative Dong Code"].astype(str)
    foreign["Nationality"] = foreign["Nationality"].astype(str).str.strip()

    native = native.dropna(subset=["Grid ID", "X", "Y", "Hour", "Stay Duration"])
    foreign = foreign.dropna(subset=["Grid ID", "Hour", "Stay Duration"])

    native["Hour"] = native["Hour"].astype(int)
    native["Stay Duration"] = native["Stay Duration"].astype(int)
    foreign["Hour"] = foreign["Hour"].astype(int)
    foreign["Stay Duration"] = foreign["Stay Duration"].astype(int)

    native["Native Population"] = native["Native Population"].fillna(0)
    foreign["Foreign Population"] = foreign["Foreign Population"].fillna(0)

    foreign_total = (
        foreign
        .groupby(["Grid ID", "Hour", "Stay Duration"], as_index=False)["Foreign Population"]
        .sum()
    )

    data = native.merge(
        foreign_total,
        on=["Grid ID", "Hour", "Stay Duration"],
        how="left"
    )

    data["Foreign Population"] = data["Foreign Population"].fillna(0)
    data["Total Population"] = data["Native Population"] + data["Foreign Population"]
    data = data[data["Total Population"] >= 0].copy()

    data, has_geo, geo_info = try_add_lat_lon(data)

    grid_lookup = (
        data
        .groupby("Grid ID", as_index=False)
        .agg({
            "X": "first",
            "Y": "first",
            "Administrative Dong Code": "first"
        })
    )

    foreign_area = foreign.copy()

    missing_area_cols = [col for col in ["X", "Y", "Administrative Dong Code"] if col not in foreign_area.columns]
    if missing_area_cols:
        foreign_area = foreign_area.drop(columns=[c for c in ["X", "Y", "Administrative Dong Code"] if c in foreign_area.columns], errors="ignore")
        foreign_area = foreign_area.merge(grid_lookup, on="Grid ID", how="left")

    foreign_area["Administrative Dong Code"] = foreign_area["Administrative Dong Code"].astype(str)
    foreign_area = foreign_area.dropna(subset=["X", "Y"])
    foreign_area, foreign_has_geo, foreign_geo_info = try_add_lat_lon(foreign_area)

    return data, native, foreign_area, has_geo, geo_info, foreign_has_geo, foreign_geo_info


# =========================================================
# 5. START APP
# =========================================================

native_source, foreign_source = get_data_sources()
data, native_raw, foreign_area, has_geo, geo_info, foreign_has_geo, foreign_geo_info = load_data(native_source, foreign_source)

st.title("Seoul Population Intelligence Dashboard")
st.caption("Spatial analytics · trend detection · nationality analysis · clustering · machine learning prediction")

st.markdown(
    """
    **Shin Do Yun**  
    **Gachon University · Industrial Engineering Student**
    """
)

st.divider()


# =========================================================
# 6. SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("Control Panel")

selected_hour = st.sidebar.slider(
    "Select hour",
    min_value=int(data["Hour"].min()),
    max_value=int(data["Hour"].max()),
    value=int(data["Hour"].median())
)

nationality_options = (
    foreign_area
    .groupby("Nationality", as_index=False)["Foreign Population"]
    .sum()
    .sort_values("Foreign Population", ascending=False)["Nationality"]
    .tolist()
)

if not nationality_options:
    nationality_options = ["Unknown"]

selected_nationality = st.sidebar.selectbox(
    "Select nationality",
    nationality_options
)

sample_size = st.sidebar.slider(
    "Map sample size",
    min_value=1000,
    max_value=100000,
    value=30000,
    step=5000
)

n_clusters = st.sidebar.slider(
    "Number of clusters",
    min_value=3,
    max_value=10,
    value=5
)

model_type = st.sidebar.selectbox(
    "Prediction model",
    ["Random Forest", "Gradient Boosting"]
)

model_sample_size = st.sidebar.slider(
    "Model sample size",
    min_value=5000,
    max_value=100000,
    value=30000,
    step=5000
)


# =========================================================
# 7. CORE AGGREGATION
# =========================================================

hourly = (
    data
    .groupby("Hour", as_index=False)
    .agg({
        "Native Population": "sum",
        "Foreign Population": "sum",
        "Total Population": "sum"
    })
    .sort_values("Hour")
)

hourly["Change From Previous Hour"] = hourly["Total Population"].diff()
hourly["Growth Rate (%)"] = hourly["Total Population"].pct_change() * 100

peak_hour = int(hourly.loc[hourly["Total Population"].idxmax(), "Hour"])
peak_population = float(hourly["Total Population"].max())

increase_df = hourly.dropna(subset=["Change From Previous Hour"])
if len(increase_df) > 0:
    max_increase_row = increase_df.sort_values("Change From Previous Hour", ascending=False).iloc[0]
    max_increase_hour = int(max_increase_row["Hour"])
    max_increase_value = float(max_increase_row["Change From Previous Hour"])
else:
    max_increase_hour = None
    max_increase_value = 0

daily_average = hourly["Total Population"].mean()
over_average = hourly[hourly["Total Population"] >= daily_average]
first_over_average_hour = int(over_average["Hour"].iloc[0]) if len(over_average) > 0 else None

top_dong = (
    data
    .groupby("Administrative Dong Code", as_index=False)["Total Population"]
    .sum()
    .sort_values("Total Population", ascending=False)
    .head(20)
)

top_grid = (
    data
    .groupby(["Grid ID", "Administrative Dong Code"], as_index=False)
    .agg({
        "X": "first",
        "Y": "first",
        "Total Population": "sum",
        "Foreign Population": "sum"
    })
    .sort_values("Total Population", ascending=False)
    .head(20)
)

top_nationality = (
    foreign_area
    .groupby("Nationality", as_index=False)["Foreign Population"]
    .sum()
    .sort_values("Foreign Population", ascending=False)
    .head(20)
)

top_area_nationality = (
    foreign_area
    .groupby(["Administrative Dong Code", "Nationality"], as_index=False)["Foreign Population"]
    .sum()
    .sort_values("Foreign Population", ascending=False)
    .head(30)
)


# =========================================================
# 8. EXECUTIVE SUMMARY
# =========================================================

st.subheader("Executive Summary")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total rows", f"{len(data):,}")
k2.metric("Unique grids", f"{data['Grid ID'].nunique():,}")
k3.metric("Peak hour", f"{peak_hour}:00", f"{peak_population:,.0f}")
k4.metric("Sharpest increase", f"{max_increase_hour}:00" if max_increase_hour is not None else "N/A", f"+{max_increase_value:,.0f}")
k5.metric("Top nationality", top_nationality.iloc[0]["Nationality"] if len(top_nationality) > 0 else "N/A")

st.caption(f"Population coordinate status: {geo_info}")
st.caption(f"Foreign population coordinate status: {foreign_geo_info}")

st.divider()


# =========================================================
# 9. TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Map",
    "2. Trend",
    "3. Area & Age",
    "4. Nationality",
    "5. Clustering",
    "6. Prediction",
    "7. Findings"
])


# =========================================================
# TAB 1. MAP
# =========================================================

with tab1:
    st.header("1. Seoul Population Map")

    current = data[data["Hour"] == selected_hour].copy()
    current_grid = (
        current
        .groupby(["Grid ID", "Administrative Dong Code"], as_index=False)
        .agg({
            "X": "first",
            "Y": "first",
            "Total Population": "sum",
            "Native Population": "sum",
            "Foreign Population": "sum"
        })
    )

    if has_geo:
        geo_df = data.groupby("Grid ID", as_index=False).agg({"lon": "first", "lat": "first"})
        current_grid = current_grid.merge(geo_df, on="Grid ID", how="left")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Population at {selected_hour}:00", f"{current_grid['Total Population'].sum():,.0f}")
    m2.metric("Active grids", f"{current_grid['Grid ID'].nunique():,}")
    m3.metric("Highest grid", f"{current_grid['Total Population'].max():,.0f}")
    m4.metric("Foreign population", f"{current_grid['Foreign Population'].sum():,.0f}")

    st.subheader(f"Spatial Distribution at {selected_hour}:00")

    if has_geo and "lon" in current_grid.columns and "lat" in current_grid.columns:
        map_df = current_grid.dropna(subset=["lon", "lat"]).copy()
        if len(map_df) > sample_size:
            map_df = map_df.sample(sample_size, random_state=42)

        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            color="Total Population",
            size="Total Population",
            hover_data=["Grid ID", "Administrative Dong Code", "Total Population", "Native Population", "Foreign Population"],
            zoom=10,
            height=720,
            title=f"Seoul 250m Grid Population Map · {selected_hour}:00",
            color_continuous_scale="Turbo",
            mapbox_style="carto-darkmatter"
        )
        fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
        st.plotly_chart(style_plotly(fig), width="stretch")
    else:
        plot_df = current_grid.copy()
        if len(plot_df) > sample_size:
            plot_df = plot_df.sample(sample_size, random_state=42)

        fig = px.scatter(
            plot_df,
            x="X",
            y="Y",
            color="Total Population",
            size="Total Population",
            hover_data=["Grid ID", "Administrative Dong Code", "Total Population"],
            height=720,
            title=f"Grid-based Population Map · {selected_hour}:00",
            color_continuous_scale="Turbo"
        )
        st.plotly_chart(style_plotly(fig), width="stretch")

    st.subheader("Top 20 Crowded Grids")
    top_current_grid = current_grid.sort_values("Total Population", ascending=False).head(20)

    fig = px.bar(
        top_current_grid,
        x="Total Population",
        y="Grid ID",
        orientation="h",
        color="Total Population",
        title="Top 20 Crowded Grids"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_plotly(fig), width="stretch")
    st.dataframe(top_current_grid, width="stretch")


# =========================================================
# TAB 2. TREND
# =========================================================

with tab2:
    st.header("2. Population Trend Detection")

    fig = px.line(
        hourly,
        x="Hour",
        y=["Native Population", "Foreign Population", "Total Population"],
        markers=True,
        title="Hourly Population Trend"
    )
    st.plotly_chart(style_plotly(fig), width="stretch")

    fig = px.bar(
        hourly,
        x="Hour",
        y="Change From Previous Hour",
        color="Change From Previous Hour",
        title="Population Change Compared to Previous Hour"
    )
    st.plotly_chart(style_plotly(fig), width="stretch")

    t1, t2, t3 = st.columns(3)
    t1.metric("Peak population hour", f"{peak_hour}:00", f"{peak_population:,.0f}")
    t2.metric("Sharpest increase hour", f"{max_increase_hour}:00" if max_increase_hour is not None else "N/A", f"+{max_increase_value:,.0f}")
    t3.metric("First above-average hour", f"{first_over_average_hour}:00" if first_over_average_hour is not None else "N/A")

    st.subheader("Hourly Trend Table")
    st.dataframe(hourly, width="stretch")


# =========================================================
# TAB 3. AREA & AGE
# =========================================================

with tab3:
    st.header("3. Area Concentration & Age Analysis")

    st.subheader("Top Administrative Dong Codes")
    fig = px.bar(
        top_dong,
        x="Total Population",
        y="Administrative Dong Code",
        orientation="h",
        color="Total Population",
        title="Top Administrative Dong Codes by Total Population"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_plotly(fig), width="stretch")
    st.dataframe(top_dong, width="stretch")

    st.subheader("Top 20 Grids by Total Population")
    fig = px.bar(
        top_grid,
        x="Total Population",
        y="Grid ID",
        orientation="h",
        color="Total Population",
        title="Top 20 Population Grids"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_plotly(fig), width="stretch")
    st.dataframe(top_grid, width="stretch")

    st.subheader("Age Group Analysis")

    age_map = {}
    for col in data.columns:
        label = detect_age_label(col)
        if label is not None:
            age_map[col] = label

    if len(age_map) == 0:
        st.warning("Age-related columns were not detected automatically.")
    else:
        age_records = []
        for col, label in age_map.items():
            value = pd.to_numeric(data[col], errors="coerce").fillna(0).sum()
            age_records.append({"Age Group": label, "Population": value})

        age_df = pd.DataFrame(age_records).groupby("Age Group", as_index=False)["Population"].sum()
        age_df["Sort"] = age_df["Age Group"].apply(age_sort_key)
        age_df = age_df.sort_values("Sort")

        largest_age = age_df.sort_values("Population", ascending=False).iloc[0]
        st.metric("Largest age group", str(largest_age["Age Group"]), f"{largest_age['Population']:,.0f}")

        fig = px.bar(
            age_df,
            x="Age Group",
            y="Population",
            color="Population",
            title="Population by Age Group"
        )
        st.plotly_chart(style_plotly(fig), width="stretch")
        st.dataframe(age_df.drop(columns=["Sort"]), width="stretch")


# =========================================================
# TAB 4. NATIONALITY
# =========================================================

with tab4:
    st.header("4. Foreign Nationality Analysis")

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Foreign rows", f"{len(foreign_area):,}")
    n2.metric("Nationality groups", f"{foreign_area['Nationality'].nunique():,}")
    n3.metric("Top nationality", top_nationality.iloc[0]["Nationality"] if len(top_nationality) > 0 else "N/A")
    n4.metric("Foreign population total", f"{foreign_area['Foreign Population'].sum():,.0f}")

    st.subheader("Top 20 Foreign Nationality Groups")
    fig = px.bar(
        top_nationality,
        x="Foreign Population",
        y="Nationality",
        orientation="h",
        color="Foreign Population",
        title="Top 20 Foreign Population Groups by Nationality"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_plotly(fig), width="stretch")
    st.dataframe(top_nationality, width="stretch")

    st.subheader("Top Area-Nationality Concentrations")
    fig = px.bar(
        top_area_nationality,
        x="Foreign Population",
        y="Administrative Dong Code",
        color="Nationality",
        orientation="h",
        title="Where Foreign Nationalities Are Concentrated"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_plotly(fig), width="stretch")
    st.dataframe(top_area_nationality, width="stretch")

    st.subheader(f"Map of Selected Nationality: {selected_nationality}")

    selected_foreign = foreign_area[
        (foreign_area["Nationality"] == selected_nationality)
        & (foreign_area["Hour"] == selected_hour)
    ].copy()

    selected_grid = (
        selected_foreign
        .groupby(["Grid ID", "Administrative Dong Code", "Nationality"], as_index=False)
        .agg({
            "X": "first",
            "Y": "first",
            "Foreign Population": "sum"
        })
        .sort_values("Foreign Population", ascending=False)
    )

    if foreign_has_geo:
        geo_foreign = foreign_area.groupby("Grid ID", as_index=False).agg({"lon": "first", "lat": "first"})
        selected_grid = selected_grid.merge(geo_foreign, on="Grid ID", how="left")

    if len(selected_grid) == 0:
        st.warning(f"No data found for {selected_nationality} at {selected_hour}:00.")
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric("Selected nationality", selected_nationality)
        s2.metric("Selected-hour population", f"{selected_grid['Foreign Population'].sum():,.0f}")
        s3.metric("Most concentrated area", selected_grid.iloc[0]["Administrative Dong Code"])

        if foreign_has_geo and "lon" in selected_grid.columns and "lat" in selected_grid.columns:
            nation_map_df = selected_grid.dropna(subset=["lon", "lat"]).copy()
            if len(nation_map_df) > sample_size:
                nation_map_df = nation_map_df.sample(sample_size, random_state=42)

            fig = px.scatter_mapbox(
                nation_map_df,
                lat="lat",
                lon="lon",
                color="Foreign Population",
                size="Foreign Population",
                hover_data=["Grid ID", "Administrative Dong Code", "Nationality", "Foreign Population"],
                zoom=10,
                height=720,
                title=f"{selected_nationality} Population Map · {selected_hour}:00",
                color_continuous_scale="Plasma",
                mapbox_style="carto-darkmatter"
            )
            fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
            st.plotly_chart(style_plotly(fig), width="stretch")
        else:
            fig = px.scatter(
                selected_grid,
                x="X",
                y="Y",
                color="Foreign Population",
                size="Foreign Population",
                hover_data=["Grid ID", "Administrative Dong Code", "Nationality", "Foreign Population"],
                height=720,
                title=f"{selected_nationality} Grid Distribution · {selected_hour}:00",
                color_continuous_scale="Plasma"
            )
            st.plotly_chart(style_plotly(fig), width="stretch")

    st.subheader("Nationality × Area Heatmap")

    top_10_nations = top_nationality.head(10)["Nationality"].tolist()
    top_15_areas = (
        foreign_area
        .groupby("Administrative Dong Code", as_index=False)["Foreign Population"]
        .sum()
        .sort_values("Foreign Population", ascending=False)
        .head(15)["Administrative Dong Code"]
        .tolist()
    )

    heat_df = foreign_area[
        foreign_area["Nationality"].isin(top_10_nations)
        & foreign_area["Administrative Dong Code"].isin(top_15_areas)
    ].copy()

    heat_pivot = heat_df.pivot_table(
        index="Administrative Dong Code",
        columns="Nationality",
        values="Foreign Population",
        aggfunc="sum",
        fill_value=0
    )

    fig = px.imshow(
        heat_pivot,
        aspect="auto",
        color_continuous_scale="Turbo",
        title="Nationality × Administrative Area Heatmap"
    )
    st.plotly_chart(style_plotly(fig), width="stretch")


# =========================================================
# TAB 5. CLUSTERING
# =========================================================

with tab5:
    st.header("5. Spatial Clustering")

    grid_base = (
        data
        .groupby("Grid ID")
        .agg({
            "X": "first",
            "Y": "first",
            "Administrative Dong Code": "first",
            "Total Population": ["mean", "max", "sum"],
            "Foreign Population": "mean"
        })
    )

    grid_base.columns = [
        "X", "Y", "Administrative Dong Code",
        "Average Population", "Maximum Population", "Cumulative Population", "Average Foreign Population"
    ]

    time_pattern = data.pivot_table(
        index="Grid ID",
        columns="Hour",
        values="Total Population",
        aggfunc="mean",
        fill_value=0
    )
    time_pattern.columns = [f"Hour_{int(col)}_Average" for col in time_pattern.columns]

    cluster_df = pd.concat([grid_base, time_pattern], axis=1).fillna(0)

    if has_geo:
        geo_df = data.groupby("Grid ID").agg({"lon": "first", "lat": "first"})
        cluster_df = cluster_df.join(geo_df, how="left")

    exclude_cols = ["Administrative Dong Code", "lon", "lat"]
    cluster_features = [col for col in cluster_df.columns if col not in exclude_cols]
    X_cluster = cluster_df[cluster_features].copy()
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=2048,
        n_init=20
    )
    cluster_df["Cluster"] = kmeans.fit_predict(X_cluster_scaled)

    cluster_summary = (
        cluster_df
        .groupby("Cluster", as_index=False)
        .agg({
            "Average Population": "mean",
            "Maximum Population": "mean",
            "Cumulative Population": "mean",
            "Average Foreign Population": "mean"
        })
        .sort_values("Average Population", ascending=False)
    )

    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary, width="stretch")

    try:
        if len(X_cluster_scaled) > 5000:
            idx = np.random.RandomState(42).choice(len(X_cluster_scaled), 5000, replace=False)
            sil = silhouette_score(X_cluster_scaled[idx], cluster_df["Cluster"].values[idx])
        else:
            sil = silhouette_score(X_cluster_scaled, cluster_df["Cluster"].values)
        st.metric("Silhouette Score", f"{sil:.4f}")
    except Exception:
        pass

    if has_geo and "lon" in cluster_df.columns and "lat" in cluster_df.columns:
        fig = px.scatter_mapbox(
            cluster_df.reset_index().dropna(subset=["lon", "lat"]),
            lat="lat",
            lon="lon",
            color="Cluster",
            size="Average Population",
            hover_data=["Grid ID", "Administrative Dong Code", "Average Population", "Average Foreign Population"],
            zoom=10,
            height=720,
            title="Spatial Clustering Map",
            mapbox_style="carto-darkmatter"
        )
        fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
        st.plotly_chart(style_plotly(fig), width="stretch")
    else:
        fig = px.scatter(
            cluster_df.reset_index(),
            x="X",
            y="Y",
            color="Cluster",
            size="Average Population",
            hover_data=["Grid ID", "Administrative Dong Code", "Average Population"],
            height=720,
            title="Grid-based Clustering Map"
        )
        st.plotly_chart(style_plotly(fig), width="stretch")


# =========================================================
# TAB 6. PREDICTION
# =========================================================

with tab6:
    st.header("6. Machine Learning Population Prediction")

    reg_df = data.sort_values(["Grid ID", "Stay Duration", "Hour"]).copy()
    reg_df["Lag 1"] = reg_df.groupby(["Grid ID", "Stay Duration"])["Total Population"].shift(1).fillna(0)
    reg_df["Lag 2"] = reg_df.groupby(["Grid ID", "Stay Duration"])["Total Population"].shift(2).fillna(0)
    reg_df["Rolling 3 Mean"] = (
        reg_df
        .groupby(["Grid ID", "Stay Duration"])["Total Population"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0)
    )

    features = [
        "X", "Y", "Hour", "Stay Duration", "Foreign Population",
        "Lag 1", "Lag 2", "Rolling 3 Mean", "Administrative Dong Code"
    ]

    if has_geo and "lon" in reg_df.columns and "lat" in reg_df.columns:
        features = [
            "lon", "lat", "X", "Y", "Hour", "Stay Duration", "Foreign Population",
            "Lag 1", "Lag 2", "Rolling 3 Mean", "Administrative Dong Code"
        ]

    target = "Total Population"
    model_df = reg_df[features + [target]].dropna().copy()

    if len(model_df) > model_sample_size:
        model_df = model_df.sample(model_sample_size, random_state=42)

    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = [col for col in features if col != "Administrative Dong Code"]
    categorical_features = ["Administrative Dong Code"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", make_ohe(), categorical_features)
        ],
        remainder="drop"
    )

    if model_type == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=160,
            max_depth=18,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    else:
        estimator = GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.06,
            max_depth=4,
            random_state=42
        )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator)
        ]
    )

    p1, p2, p3 = st.columns(3)
    p1.metric("Selected model", model_type)
    p2.metric("Training rows", f"{len(X_train):,}")
    p3.metric("Test rows", f"{len(X_test):,}")

    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    r1, r2_col, r3 = st.columns(3)
    r1.metric("MAE", f"{mae:,.2f}")
    r2_col.metric("RMSE", f"{rmse:,.2f}")
    r3.metric("R²", f"{r2:.4f}")

    result_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })
    result_df["Error"] = result_df["Actual"] - result_df["Predicted"]
    result_df["Absolute Error"] = result_df["Error"].abs()

    plot_result = result_df.sample(min(5000, len(result_df)), random_state=42)

    fig = px.scatter(
        plot_result,
        x="Actual",
        y="Predicted",
        opacity=0.55,
        title="Actual Population vs Predicted Population"
    )

    min_v = min(plot_result["Actual"].min(), plot_result["Predicted"].min())
    max_v = max(plot_result["Actual"].max(), plot_result["Predicted"].max())
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Perfect Prediction"))
    st.plotly_chart(style_plotly(fig), width="stretch")

    fig = px.histogram(
        result_df,
        x="Error",
        nbins=60,
        title="Prediction Error Distribution"
    )
    st.plotly_chart(style_plotly(fig), width="stretch")

    if model_type == "Random Forest":
        try:
            feature_names = model.named_steps["preprocess"].get_feature_names_out()
            importances = model.named_steps["model"].feature_importances_
            importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            importance_df = importance_df.sort_values("Importance", ascending=False).head(25)

            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                title="Feature Importance"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_plotly(fig), width="stretch")
            st.dataframe(importance_df, width="stretch")
        except Exception:
            st.info("Feature importance could not be calculated.")


# =========================================================
# TAB 7. FINDINGS
# =========================================================

with tab7:
    st.header("7. Findings & Limitations")

    top_nation_text = top_nationality.iloc[0]["Nationality"] if len(top_nationality) > 0 else "N/A"
    top_area = top_dong.iloc[0]["Administrative Dong Code"] if len(top_dong) > 0 else "N/A"
    top_grid_id = top_grid.iloc[0]["Grid ID"] if len(top_grid) > 0 else "N/A"

    st.markdown(
        f"""
        ## Key Findings

        - The highest population appears at **{peak_hour}:00**.
        - The strongest population increase occurs at **{max_increase_hour}:00**.
        - The first above-average population hour is **{first_over_average_hour}:00**.
        - The most crowded administrative dong code is **{top_area}**.
        - The most crowded 250m grid is **{top_grid_id}**.
        - The largest foreign nationality group is **{top_nation_text}**.

        ---

        ## Project Summary

        This project analyzes Seoul 250m grid-level stay population data through an interactive dashboard.

        The dashboard includes:

        1. Seoul grid-level spatial visualization  
        2. Hourly population trend detection  
        3. Administrative area and grid concentration analysis  
        4. Age group analysis  
        5. Foreign nationality distribution analysis  
        6. Area-nationality concentration analysis  
        7. KMeans-based spatial clustering  
        8. Machine learning-based population prediction  

        ---

        ## Limitations

        - The dataset appears to represent a limited time period, so the model should be interpreted as a short-term analytical model.
        - External variables such as weather, weekday/weekend, holidays, subway accessibility, commercial districts, and events are not included.
        - If more time-series data is added, this project can be extended into real population forecasting.

        ---

        ## Portfolio Value

        This project demonstrates practical data analytics skills including preprocessing, EDA, visualization, geospatial analysis, clustering, machine learning, and dashboard development.
        """
    )
