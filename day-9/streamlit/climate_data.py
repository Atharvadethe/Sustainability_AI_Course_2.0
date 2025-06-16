import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the climate data once
climate = pd.read_csv(r'D:\PYTHON\Edunet2.0\day-9\climate_data(in).csv')

st.title('Climate Energy Consumption Predictor')

# Prepare features and target
target_col = 'Energy Consumption'
feature_cols = [col for col in climate.columns if col != target_col]
X = climate[feature_cols]
y = climate[target_col]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build and train the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

st.header('Enter Climate Features to Predict Energy Consumption')
user_input = {}
for col in feature_cols:
    val = st.number_input(f"{col}", float(climate[col].min()), float(climate[col].max()), float(climate[col].mean()))
    user_input[col] = val

if st.button('Predict Energy Consumption'):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Energy Consumption: {prediction[0][0]:.2f}")

st.markdown('---')
st.subheader('Model Performance on Test Data')
loss, mae = model.evaluate(X_test, y_test, verbose=0)
st.write(f"Test Loss: {loss:.4f}")
st.write(f"Test MAE: {mae:.4f}")
y_pred = model.predict(X_test)
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(range(len(y_test)), y_test.values, label='Actual')
ax.plot(range(len(y_pred)), y_pred.flatten(), label='Predicted')
ax.set_title('Actual vs Predicted Energy Consumption')
ax.set_xlabel('Sample')
ax.set_ylabel('Energy Consumption')
ax.legend()
st.pyplot(fig)