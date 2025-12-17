import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
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
Bhutan has made remarkable progress toward malaria elimination,  
achieving **zero indigenous cases since 2021**.

This dashboard visualizes historical indicators and provides  
**simple future projections based on past trends**.
""")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/malaria_indicators_btn.csv")

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
    "Select Year (Map Highlight)",
    int(filtered_df["year"].min()),
    int(filtered_df["year"].max()),
    int(filtered_df["year"].max())
)

forecast_years = st.sidebar.slider(
    "Years to Forecast",
    1, 10, 5
)

# --------------------------------------------------
# KPI
# --------------------------------------------------
latest_row = filtered_df.iloc[-1]
st.metric(
    label=f"Latest Value ({int(latest_row['year'])})",
    value=f"{latest_row['value']:.2f}"
)

# --------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Line Trend")
    fig_line = px.line(
        filtered_df,
        x="year",
        y="value",
        markers=True
    )
    st.plotly_chart(fig_line, width="stretch")

with col2:
    st.subheader("Bar Chart")
    fig_bar = px.bar(
        filtered_df,
        x="year",
        y="value"
    )
    st.plotly_chart(fig_bar, width="stretch")

# --------------------------------------------------
# HISTOGRAM
# --------------------------------------------------
st.subheader("Histogram")
fig_hist = px.histogram(
    filtered_df,
    x="value",
    nbins=10
)
st.plotly_chart(fig_hist, width="stretch")

# --------------------------------------------------
# FUTURE PREDICTION (NUMPY)
# --------------------------------------------------
st.subheader("🔮 Future Projection")

# Linear trend using numpy (cloud-safe)
x = filtered_df["year"].values
y = filtered_df["value"].values

coef = np.polyfit(x, y, 1)
trend = np.poly1d(coef)

future_years = np.arange(
    int(x.max()) + 1,
    int(x.max()) + forecast_years + 1
)

future_values = trend(future_years)

forecast_df = pd.DataFrame({
    "year": np.concatenate([x, future_years]),
    "value": np.concatenate([y, future_values]),
    "type": ["Observed"] * len(x) + ["Predicted"] * len(future_years)
})

fig_forecast = px.line(
    forecast_df,
    x="year",
    y="value",
    color="type",
    markers=True
)
st.plotly_chart(fig_forecast, width="stretch")

# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------
st.subheader("Data Table")
st.dataframe(filtered_df)

# --------------------------------------------------
# MAP (YEAR-WISE HIGHLIGHT)
# --------------------------------------------------
st.subheader("🗺️ Bhutan District Map")

geojson = json.load(open("data/bhutan_districts.json"))

year_value = filtered_df[filtered_df["year"] == selected_year]["value"].mean()
intensity = 50 if np.isnan(year_value) else min(255, int(year_value * 10))

layer = pdk.Layer(
    "GeoJsonLayer",
    geojson,
    filled=True,
    stroked=True,
    get_fill_color=f"[255, {255-intensity}, {255-intensity}, 140]",
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

st.caption("District coloring is illustrative. Replace with district-level malaria data when available.")

