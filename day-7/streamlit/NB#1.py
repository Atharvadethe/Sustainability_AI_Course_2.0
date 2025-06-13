import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# For SMOTE
from imblearn.over_sampling import SMOTE

st.title("Ecosystem Health Classification with Naive Bayes & SMOTE")

# Load the dataset
data = pd.read_csv(r'D:\PYTHON\Edunet2.0\day-7\ecosystem_data.csv')
st.write("### Dataset Preview", data.head())

# Encode target
data['ecosystem_health'] = data['ecosystem_health'].map({'healthy': 1, 'at risk': 0, 'degraded': 2})

# Features and target
X = data[['water_quality','air_quality_index','biodiversity_index','vegetation_cover','soil_ph']]
y = data['ecosystem_health']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Results Without SMOTE")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'At Risk', 'Degraded'], 
            yticklabels=['Healthy', 'At Risk', 'Degraded'], ax=ax)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (No SMOTE)')
st.pyplot(fig)

st.text("Classification Report (No SMOTE):")
st.text(classification_report(y_test, y_pred, target_names=['Healthy', 'At Risk', 'Degraded']))

# SMOTE
st.subheader("Results With SMOTE (Balanced Classes)")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

gnb_res = GaussianNB()
gnb_res.fit(X_train_res, y_train_res)
y_pred_res = gnb_res.predict(X_test_res)
accuracy_res = accuracy_score(y_test_res, y_pred_res)

st.write(f"Accuracy after SMOTE: {accuracy_res * 100:.2f}%")

# Confusion matrix after SMOTE
cm_res = confusion_matrix(y_test_res, y_pred_res)
fig2, ax2 = plt.subplots()
sns.heatmap(cm_res, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'At Risk', 'Degraded'], 
            yticklabels=['Healthy', 'At Risk', 'Degraded'], ax=ax2)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (With SMOTE)')
st.pyplot(fig2)

st.text("Classification Report (With SMOTE):")
st.text(classification_report(y_test_res, y_pred_res, target_names=['Healthy', 'At Risk', 'Degraded']))

# User input for prediction
st.subheader("Predict Ecosystem Health (Enter Feature Values)")
wq = st.number_input("Water Quality Index", min_value=0.0, max_value=100.0, value=50.0)
aqi = st.number_input("Air Quality Index", min_value=0.0, max_value=500.0, value=100.0)
biodiversity = st.number_input("Biodiversity Index", min_value=0.0, max_value=100.0, value=50.0)
veg_cover = st.number_input("Vegetation Cover Index", min_value=0.0, max_value=100.0, value=50.0)
soil_ph = st.number_input("Soil pH Index", min_value=0.0, max_value=14.0, value=7.0)

user_input = np.array([[wq, aqi, biodiversity, veg_cover, soil_ph]])

if st.button("Predict (No SMOTE)"):
    pred = gnb.predict(user_input)[0]
    if pred == 1:
        st.success("Predicted: Healthy")
    elif pred == 0:
        st.warning("Predicted: At Risk")
    elif pred == 2:
        st.error("Predicted: Degraded")

if st.button("Predict (With SMOTE)"):
    pred_res = gnb_res.predict(user_input)[0]
    if pred_res == 1:
        st.success("Predicted (SMOTE): Healthy")
    elif pred_res == 0:
        st.warning("Predicted (SMOTE): At Risk")
    elif pred_res == 2:
        st.error("Predicted (SMOTE): Degraded")