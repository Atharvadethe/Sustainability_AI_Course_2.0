import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st

st.title("Weather Classification: Decision Tree, Random Forest, Logistic Regression")

# Load data
df = pd.read_csv('D:\PYTHON\Edunet2.0\day-6\Weather Data.csv')
st.write("Dataset Preview:", df.head())

# Drop rows with missing values
df = df.dropna()

# Set target and features
target = 'Weather'
X = df.drop(columns=[target, 'Date/Time'])
y = df[target]

# Encode categorical target
y_encoded = y.astype('category').cat.codes
label_mapping = dict(enumerate(y.astype('category').cat.categories))

# Encode features if needed
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

results = {}

st.subheader("Model Accuracies")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': acc}
    st.write(f"{name} - Accuracy: {acc:.2f}")

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
st.success(f"Best model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.2f})")

# Save the best model, features, and label mapping
joblib.dump(best_model, 'best_weather_model.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')
joblib.dump(label_mapping, 'label_mapping.pkl')

# Show classification report for best model
y_pred_best = best_model.predict(X_test)
st.subheader("Classification Report for Best Model")
all_labels = list(label_mapping.keys())
all_names = list(label_mapping.values())
st.text(classification_report(
    y_test, y_pred_best, labels=all_labels, target_names=all_names, zero_division=0
))