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

st.title("📊 Bhutan Malaria Indicators Dashboard")

st.write("Visualizing malaria indicators from WHO dataset.")

# -------------------------------------------
# Clean and prepare columns
# -------------------------------------------

# Rename important columns to simple names
df = df.rename(columns={
    "GHO (DISPLAY)": "indicator_name",
    "YEAR (DISPLAY)": "year",
    "Numeric": "value_num"
})

# Ensure value_num is numeric
df["value_num"] = pd.to_numeric(df["value_num"], errors="coerce")

# -------------------------------------------
# Sidebar selection
# -------------------------------------------
indicator_list = df["indicator_name"].dropna().unique()
year_list = sorted(df["year"].dropna().unique())

selected_indicator = st.sidebar.selectbox("Select Indicator", indicator_list)
selected_year = st.sidebar.selectbox("Select Year", year_list)

# -------------------------------------------
# Filter Data
# -------------------------------------------
filtered_df = df[
    (df["indicator_name"] == selected_indicator) &
    (df["year"] == selected_year)
]

st.subheader(f"{selected_indicator} — {selected_year}")
st.dataframe(filtered_df)

# -------------------------------------------
# Plot trend graph
# -------------------------------------------
plot_df = df[df["indicator_name"] == selected_indicator]

fig = px.line(
    plot_df,
    x="year",
    y="value_num",
    title=f"Trend Over Time: {selected_indicator}",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------
# Summary statistics
# -------------------------------------------
st.subheader("📌 Summary Statistics")
st.write(plot_df["value_num"].describe())
