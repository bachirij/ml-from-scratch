"""
02_layout.py
------------
Objective: introduce the three main Streamlit layout primitives:
sidebar, columns, and expander.

Key concepts:
- st.sidebar: a fixed left panel for filters and configuration, separate from the main content area
- st.columns: splits the main content area into N horizontal columns
- st.expander: a collapsible section to hide secondary content
"""

import streamlit as st

# Page configuration (must be the first Streamlit call in the script)
# Controls browser tab title, page layout, and sidebar default state
st.set_page_config(page_title="My Basic Streamlit App", layout="wide")

# Print the app title
st.title("My Basic Streamlit App")

# Print a header
st.header("My first widget", divider='rainbow')

# Sidebar with a select box widget:
with st.sidebar:
    country = st.selectbox("Select a country:", ("France", "Germany", "Spain"))

# Create 2 columns
col1, col2 = st.columns(2)

# In the first column (left)
with col1:
    # Print a fictive value (e.g. national electricity consumption in MW)
    st.metric(label="National Load", value="67 GW")

# In the second column (right)
with col2:
    st.metric(label="Peak Demand", value="97 GW")
    
# Add an expander (a collapsible section to hide secondary content)
with st.expander("See more details"):
    st.write("This is an example of an expander with more details.")