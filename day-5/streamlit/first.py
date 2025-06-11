import streamlit as st
import pandas as pd

st.write("Hello, Streamlit!")
st.write("This is a simple Streamlit app.")

# Create a simple DataFrame
df = pd.DataFrame({
    'Column 1': [1, 2, 3, 4, 5],
    'Column 2': ['A', 'B', 'C', 'D', 'E']
})
st.write("Here is a simple DataFrame:")
st.line_chart(df)

# Add a title
st.title("My First Streamlit App")


