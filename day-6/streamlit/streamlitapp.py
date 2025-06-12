import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load and prepare the data
data = pd.read_csv(r'D:\PYTHON\Edunet2.0\day-6\Classifying waste_dataset.csv')

# Encode the target column (Organic, Plastic, Metal -> 0, 1, 2)
le = LabelEncoder()
data['Type'] = le.fit_transform(data['Type'])

# Features and target
X = data.drop('Type', axis=1)
y = data['Type']

# Train the Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X, y)

# Streamlit UI
st.title("Waste Type Predictor")

# Input fields
weight = st.number_input("Weight", min_value=0.0, max_value=2.0 , step=0.01)
color = st.selectbox("Color", [0, 1])
texture = st.selectbox("Texture", [0, 1])
odor = st.selectbox("Odor", [0, 1])

# Prediction
if st.button("Predict"):
    input_data = np.array([[weight, color, texture, odor]])
    prediction = model.predict(input_data)[0]
    label = le.inverse_transform([prediction])[0]
    st.subheader(f"The predicted type is: {label}")
