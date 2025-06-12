# import streamlit as st
# import pandas as pd

# st.write("Hello  Streamlit APP")
# df = pd.read_csv('Weather Data.csv')

# st.write(df)
# st.line_chart(df)


import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("♻ Waste Classification App")
st.write("Enter waste features to predict the type.")

# Sample input fields (customize based on your dataset's features)
feature_1 = st.number_input("Weight")
feature_2 = st.number_input("Color")
feature_3 = st.number_input("Texture")
feature_4 = st.number_input("Odour")


# You can add more depending on your dataset

# Prediction button
if st.button("Predict Waste Type"):
    input_data = np.array([[organic, metal, plastic, feature_4]])
    prediction = model.predict(input_data)[0]
    
    # Decode the label if needed
    st.success(f"Predicted Waste Type: {prediction}")
