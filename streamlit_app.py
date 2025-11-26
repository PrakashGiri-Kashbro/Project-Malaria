import streamlit as st
import pandas as pd
import pydeck as pdk
import json

# Load district-level malaria data (2015–2023) that you manually collected from Google
district_df = pd.read_csv("data/bhutan_district_malaria_2015_2023.csv")

# Select year
year_list = sorted(district_df["year"].unique())
selected_year = st.sidebar.selectbox("Select Year", year_list)

year_df = district_df[district_df["year"] == selected_year]

# Load GeoJSON
with open("data/bhutan_districts.json","r") as f:
    geojson = json.load(f)

# Merge district values into GeoJSON
for feature in geojson["features"]:
    district_name = feature["properties"]["DTN"]
    match = year_df[year_df["district"] == district_name]
    if not match.empty:
        value = float(match["cases"].values[0])
        feature["properties"]["value"] = value
    else:
        feature["properties"]["value"] = 0

# Pydeck layer with color based on cases
layer = pdk.Layer(
    "GeoJsonLayer",
    geojson,
    filled=True,
    stroked=True,
    get_fill_color="""
    [
        255 * (properties.value /  max(1, properties.value)),
        50,
        50,
        150
    ]
    """,
)

view_state = pdk.ViewState(
    longitude=90.4,
    latitude=27.5,
    zoom=7
)

st.header("Bhutan District Map (Malaria Cases)")
st.write(f"Showing malaria case distribution for the year: {selected_year}")

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": "<b>{DTN}</b><br/>Cases: {value}"}
    )
)
