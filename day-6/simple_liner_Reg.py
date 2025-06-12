import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"day-6\appliance_energy.csv")


X = df[['Temperature (°C)']]
y = df['Energy Consumption (kWh)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

st.write("Test Actual vs Predicted:")
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
st.write(result_df.head())


plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.legend()
st.pyplot(plt)

#calculate the R-squared score
r_squared = model.score(X_test, y_test)
st.write("R-squared score:", r_squared)

#save the model
import joblib
joblib.dump(model, r'D:\PYTHON\Edunet2.0\day-6\simple_linear_model.pkl')
