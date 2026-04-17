"""
03_charts.py
------------
Objective: introduce Plotly integration in Streamlit for interactive time series visualization.

Key concepts:
- go.Figure with go.Scatter for time series traces
- st.plotly_chart with use_container_width=True for responsive rendering
- sidebar selectbox to filter the dataframe and trigger a rerun with updated data

"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page configuration (must be the first Streamlit call in the script)
# Controls browser tab title, page layout, and sidebar default state
st.set_page_config(page_title="My Basic Streamlit App", layout="wide")

# Print the app title + header
st.title("My Basic Streamlit App")
st.header("My first chart", divider='rainbow')

# Synthetic hourly demand dataset: 7 days at 1-hour resolution, values in MW
df = pd.DataFrame({"demand": np.random.rand(7*24) * 1e4})
df.index = pd.date_range(start="2024-01-01", periods=7*24, freq="h")

# Sidebar: time window selector
# Changing this value triggers a full rerun and updates the chart
with st.sidebar:
    num_days = st.selectbox("Select the number of days to print:", (1, 3, 7))

# Filter to the selected time window (num_days * 24 hourly rows)
df_filtered = df.iloc[:num_days*24]

# Build the Plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_filtered.index, y = df_filtered["demand"]))
fig.update_layout(
    title=f"Load curve of the past {num_days}-days",
    xaxis_title="Date",
    yaxis_title="Load (MW)"
)

st.plotly_chart(fig, width="stretch")