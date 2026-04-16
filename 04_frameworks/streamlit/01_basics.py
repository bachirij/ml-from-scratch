"""
01_basics.py
------------
Objective: introduce the core Streamlit execution model through three primitives:
a display call (st.title, st.write), an input widget (st.slider), and a button.

Key concept to observe: every widget interaction triggers a full top-to-bottom
rerun of this script. The slider value is re-read on every rerun. The button
returns True only on the rerun triggered by the click, the next interaction
resets it to False.
"""

import streamlit as st

# Page configuration (must be the first Streamlit call in the script)
# Controls browser tab title, page layout, and sidebar default state
st.set_page_config(page_title="My Basic Streamlit App", layout="wide")

# Print the app title
st.title("My Basic Streamlit App")

# Print a header
st.header("My first widget", divider='rainbow')

# Print text as a markdown
st.markdown("Get the squared value:")

# Slider widget: returns the current value selected by the user
# On every rerun, Streamlit re-reads this value from its internal state
x = st.slider("Choose an x value between 1 and 10", 1, 10)

# Compute the square of x 
x_2 = x**2

# Print the result (in blue)
st.write("The value squared value of :blue[**x**] is:", x_2)

# Button: returns True only on the single rerun triggered by the click
# On the next rerun (e.g. moving the slider), it resets to False
# To persist a click across reruns, use st.session_state instead
if st.button(label="Button"):
    st.write("Button clicked!")
