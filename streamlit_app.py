import streamlit as st
import pandas as pd
import plotly.express as px

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("data/malaria_indicators_btn.csv")
    return df

df = load_data()

st.title("Bhutan Malaria Indicators Dashboard")
st.write("This simple dashboard shows malaria-related indicators over the years.")

# Rename columns for easier use
df = df.rename(columns={
    "GHO (DISPLAY)": "indicator_name",
    "YEAR (DISPLAY)": "year",
    "Numeric": "value_num"
})

# convert numeric column properly
df["value_num"] = pd.to_numeric(df["value_num"], errors="coerce")

# Sidebar – only select indicator
indicator_list = df["indicator_name"].dropna().unique()

selected_indicator = st.sidebar.selectbox(
    "Select an Indicator",
    indicator_list
)

# Filter data
filtered_df = df[df["indicator_name"] == selected_indicator]

st.subheader(f"Indicator: {selected_indicator}")

# Bar chart
st.write("### Bar Chart")
fig_bar = px.bar(
    filtered_df,
    x="year",
    y="value_num",
    title=f"{selected_indicator} Over Years"
)
st.plotly_chart(fig_bar, use_container_width=True)

# Line chart
st.write("### Line Chart")
fig_line = px.line(
    filtered_df,
    x="year",
    y="value_num",
    markers=True
)
st.plotly_chart(fig_line, use_container_width=True)

# Show data table
st.write("### Data Table")
st.dataframe(filtered_df)

st.header("Bhutan District Map")

import pydeck as pdk
import json
import pandas as pd

geojson = json.load(open("data/bhutan_districts.geojson"))

layer = pdk.Layer(
    "GeoJsonLayer",
    geojson,
    stroked=True,
    filled=True,
    get_fill_color="[255, 0, 0, 100]",
)

view_state = pdk.ViewState(
    latitude=27.5,
    longitude=90.4,
    zoom=7
)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state
    )
)

