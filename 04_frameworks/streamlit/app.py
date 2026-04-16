# Structure of a Streamlit app
import streamlit as st
import pandas as pd
import numpy as np

# Print the app title
st.title("My app")

# Print a header
st.header("This is the first header with a raindow divider", divider='rainbow')

# Print text
st.write("Hello world!")

# Print text as markdown
st.markdown("This is created using st.markdown().")

# Change the display with columns
col1, col2 = st.columns(2)

# Print in col1
with col1:
    # Slider (interactivity): the user selects a value between 1 and 10
    x = st.slider("Choose an x value", 1, 10)

# Print in col2
with col2:
    # Return the user selected value with x printed is red
    st.write("The value of :red[**x**] is:", x)



# In the terminal, type: streamlit run app.py