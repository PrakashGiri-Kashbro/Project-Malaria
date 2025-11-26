import streamlit as st
import pandas as pd
import pydeck as pdk
import json

# Load district-level data from the correct file
district_df = pd.read_csv("data/malaria_indicators_btn.csv")

# Rename columns if needed
district_df = district_df.rename(columns={
    "DISTRICT": "district",
    "YEAR": "year",
    "CASES": "cases"
})

# Sidebar year selector
year_list = sorted(district_df["year"].unique())
selected_year = st.sidebar.selectbox("Select Year", year_list)

year_df = district_df[district_df["year"] == selected_year]

# Load GeoJSON
with open("data/bhutan_districts.json","r") as f:
    geojson = json.load(f)

# Merge the values into GeoJSON
for feature in geojson["features"]:
    district_name = feature["properties"]["DTN"]
    match = year_df[year_df["district"] == district_name]
    if not match.empty:
        feature["properties"]["value"] = float(match["cases"].values[0])
    else:
        feature["properties"]["value"] = 0

# Pydeck Layer
layer = pdk.Layer(
    "GeoJsonLayer",
    geojson,
    filled=True,
    stroked=True,
    get_fill_color="[255 * (properties.value > 0), 0, 0, 150]"
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
