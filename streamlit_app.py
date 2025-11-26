import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------
# Load Data
# -------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/malaria_indicators_btn.csv")
    return df

df = load_data()

st.title("🦟 Bhutan Malaria Indicators Dashboard")
st.markdown("Explore malaria indicators over the years using WHO dataset.")

# -------------------------------------------
# Clean Column Names
# -------------------------------------------
df = df.rename(columns={
    "GHO (DISPLAY)": "indicator_name",
    "YEAR (DISPLAY)": "year",
    "Numeric": "value_num"
})

# Convert numeric column
df["value_num"] = pd.to_numeric(df["value_num"], errors="coerce")

# -------------------------------------------
# Sidebar: Indicator Selection ONLY
# -------------------------------------------
indicator_list = sorted(df["indicator_name"].dropna().unique())

selected_indicator = st.sidebar.selectbox(
    "Select Indicator",
    indicator_list
)

# -------------------------------------------
# Filter Data for Selected Indicator
# -------------------------------------------
filtered_df = df[df["indicator_name"] == selected_indicator]

st.subheader(f"📌 {selected_indicator}")

# -------------------------------------------
# Beautiful Bar Graph
# -------------------------------------------
fig = px.bar(
    filtered_df,
    x="year",
    y="value_num",
    text="value_num",
    title=f"{selected_indicator} (Across Years)",
)

fi
