"""
dashboard.py - Entry point for a multi-page Streamlit application.

Responsibilities:
- Page configuration (must be first Streamlit call)
- Sidebar navigation using st.session_state
- Routing to the active tab's render() function

Does not load data or compute anything directly.
All data loading belongs in utils/data_loader.py.
All rendering belongs in tabs/*.py.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration - must be the first Streamlit call in the script
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="My App",        # Browser tab title
    page_icon=None,             # Optional: emoji string or Path to .ico
    layout="wide",              # "wide" | "centered"
    initial_sidebar_state="expanded",  # "expanded" | "collapsed"
)

# ---------------------------------------------------------------------------
# Navigation state
# Initialize once - persists across reruns for the duration of the session
# ---------------------------------------------------------------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "overview"   # Default page key

# ---------------------------------------------------------------------------
# Sidebar - navigation + metadata
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("My App")
    st.caption("Short description of the app")
    st.divider()

    # Page registry - key: internal identifier, value: display label
    # Add or remove pages here; routing below updates automatically
    pages = {
        "overview": "Overview",
        "analysis": "Analysis",
        "settings": "Settings",
    }

    # Render one button per page
    # Clicking sets the active_page key and triggers a rerun
    for key, label in pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.active_page = key

    st.divider()
    # Optional: data source credits, model info, version
    st.caption("Data: ...")
    st.caption("Model: ...")

# ---------------------------------------------------------------------------
# Routing - import and render only the active page
# Lazy imports avoid loading all tab modules on every rerun
# ---------------------------------------------------------------------------
page = st.session_state.active_page

if page == "overview":
    from tabs.overview import render
    render()
elif page == "analysis":
    from tabs.analysis import render
    render()
elif page == "settings":
    from tabs.settings import render
    render()