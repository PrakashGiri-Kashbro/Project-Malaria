import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import pydeck as pdk
import json

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Bhutan Malaria Dashboard",
    layout="wide"
)

st.title("🇧🇹 Bhutan Malaria Indicators Dashboard")

st.markdown("""
Bhutan has made significant progress toward eliminating malaria, achieving **zero indigenous cases since 2021**.
This dashboard visualizes historical trends and provides **simple future projections** based on past data.
""")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/malaria_indicators_btn.csv")
    return df

df = load_data()

# Rename columns
df = df.rename(columns={
    "GHO (DISPLAY)": "indicator",
    "YEAR (DISPLAY)": "year",
    "Numeric": "value"
})

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["year", "value"])

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Controls")

indicator_list = sorted(df["indicator"].unique())
selected_indicator = st.sidebar.selectbox(
    "Select Indicator",
    indicator_list
)

filtered_df = df[df["indicator"] == selected_indicator].sort_values("year")

selected_year = st.sidebar.slider(
    "Select Year (for map)",
    int(filtered_df["year"].min()),
    int(filtered_df["year"].max()),
    int(filtered_df["year"].max())
)

forecast_years = st.sidebar.slider(
    "Forecast Years Into Future",
    1, 10, 5
)

# --------------------------------------------------
# KPI
# --------------------------------------------------
st.subheader(f"📊 Indicator: {selected_indicator}")
latest_value = filtered_df.iloc[-1]["value"]
latest_year = int(filtered_df.iloc[-1]["year"])

st.metric(
    label=f"Latest Value ({latest_year})",
    value=f"{latest_value:,.2f}"
)

# --------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Line Chart")
    fig_line = px.line(
        filtered_df,
        x="year",
        y="value",
        markers=True
    )
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.markdown("### Bar Chart")
    fig_bar = px.bar(
        filtered_df,
        x="year",
        y="value"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------
# HISTOGRAM
# --------------------------------------------------
st.markdown("### Histogram of Values")
fig_hist = px.histogram(
    filtered_df,
    x="value",
    nbins=10
)
st.plotly_chart(fig_hist, use_container_width=True)

# --------------------------------------------------
# PREDICTION MODEL
# --------------------------------------------------
st.markdown("### 🔮 Future Prediction (Linear Trend)")

X = filtered_df[["year"]]
y = filtered_df["value"]

model = LinearRegression()
model.fit(X, y)

future_years = np.arange(
    int(filtered_df["year"].max()) + 1,
    int(filtered_df["year"].max()) + forecast_years + 1
)

future_df = pd.DataFrame({"year": future_years})
future_df["predicted_value"] = model.predict(future_df[["year"]])

forecast_plot_df = pd.concat([
    filtered_df[["year", "value"]].rename(columns={"value": "cases"}),
    future_df.rename(columns={"predicted_value": "cases"})
])

forecast_plot_df["type"] = (
    ["Actual"] * len(filtered_df) +
    ["Predicted"] * len(future_df)
)

fig_forecast = px.line(
    forecast_plot_df,
    x="year",
    y="cases",
    color="type",
    markers=True
)
st.plotly_chart(fig_forecast, use_container_width=True)

# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------
st.markdown("### Data Table")
st.dataframe(filtered_df)

# --------------------------------------------------
# MAP (YEAR-WISE HIGHLIGHT)
# --------------------------------------------------
st.header("🗺️ Bhutan District Map (Year-wise Indicator)")

geojson = json.load(open("data/bhutan_districts.json"))

# NOTE:
# If district-level malaria data becomes available,
# replace this simulated scaling logic
year_value = filtered_df[filtered_df["year"] == selected_year]["value"].mean()
fill_intensity = min(255, int(year_value * 10)) if not np.isnan(year_value) else 50

layer = pdk.Layer(
    "GeoJsonLayer",
    geojson,
    stroked=True,
    filled=True,
    get_fill_color=f"[255, {255 - fill_intensity}, {255 - fill_intensity}, 140]",
    get_line_color=[0, 0, 0],
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=27.5,
    longitude=90.4,
    zoom=7
)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": f"Year: {selected_year}"}
    )
)

st.caption("⚠️ District coloring is illustrative. Replace with real district-level data when available.")

