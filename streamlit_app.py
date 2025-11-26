import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------
# Load Data
# -------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/malaria_indicators_btn.csv")
    return df

df = load_data()

st.set_page_config(page_title="Bhutan Malaria Dashboard", layout="wide")

st.title("🦟 Bhutan Malaria Indicators Dashboard")
st.markdown("A modern dashboard to explore malaria indicators over time using WHO data.")

# -------------------------------------------
# Clean Columns
# -------------------------------------------
df = df.rename(columns={
    "GHO (DISPLAY)": "indicator_name",
    "YEAR (DISPLAY)": "year",
    "Numeric": "value_num"
})

df["value_num"] = pd.to_numeric(df["value_num"], errors="coerce")

# -------------------------------------------
# Sidebar Selection
# -------------------------------------------
indicator_list = sorted(df["indicator_name"].dropna().unique())

selected_indicator = st.sidebar.selectbox(
    "📝 Select Indicator",
    indicator_list
)

# Load filtered data
filtered_df = df[df["indicator_name"] == selected_indicator]

# -------------------------------------------
# KPI Metrics (Top Row)
# -------------------------------------------
col1, col2, col3 = st.columns(3)

latest_year = filtered_df["year"].max()
latest_value = filtered_df[filtered_df["year"] == latest_year]["value_num"].values[0]

min_year = filtered_df["year"].min()
min_value = filtered_df[filtered_df["year"] == min_year]["value_num"].values[0]

avg_value = round(filtered_df["value_num"].mean(), 2)

col1.metric("📌 Latest Value", f"{latest_value}", f"Year {latest_year}")
col2.metric("📉 Earliest Value", f"{min_value}", f"Year {min_year}")
col3.metric("📊 Average", avg_value)

st.markdown("---")

# -------------------------------------------
# Bar Chart (Beautiful)
# -------------------------------------------
st.subheader(f"📊 Bar Chart: {selected_indicator}")

fig_bar = px.bar(
    filtered_df,
    x="year",
    y="value_num",
    text="value_num",
    title=f"{selected_indicator} Across Years",
    color="value_num",
    color_continuous_scale="Viridis",
)

fig_bar.update_traces(textposition='outside')
fig_bar.update_layout(
    template="plotly_white",
    xaxis_title="Year",
    yaxis_title="Value",
    height=500
)

st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------------------
# Line Chart with Marker
# -------------------------------------------
st.subheader("📈 Trend Line")

fig_line = px.line(
    filtered_df,
    x="year",
    y="value_num",
    markers=True,
    title=f"Trend Over Time — {selected_indicator}",
    color_discrete_sequence=["#2E86C1"]
)

fig_line.update_layout(
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_line, use_container_width=True)

# -------------------------------------------
# Animated Bar Chart
# -------------------------------------------
st.subheader("🎬 Animated Yearly Comparison (Across All Indicators)")

fig_anim = px.bar(
    df,
    x="indicator_name",
    y="value_num",
    color="indicator_name",
    animation_frame="year",
    title="Animated Indicator Comparison Over Time",
    height=600
)

fig_anim.update_layout(template="plotly_white")
st.plotly_chart(fig_anim, use_container_width=True)

# -------------------------------------------
# Data Table
# -------------------------------------------
with st.expander("📄 View Data Table"):
    st.dataframe(filtered_df)
