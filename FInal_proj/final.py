import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Load data
DATA_PATH = os.path.join('archive', 'crop_yield.csv')
df = pd.read_csv(DATA_PATH)

# Data cleaning
for col in ['Crop', 'Season', 'State']:
    df[col] = df[col].astype(str).str.strip()

# Features and target
target = 'Yield'
features = ['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

# Encode categorical features
cat_features = ['Crop', 'Season', 'State']
encoders = {col: LabelEncoder().fit(df[col]) for col in cat_features}
for col in cat_features:
    df[col] = encoders[col].transform(df[col])

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_options = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42)
}

st.sidebar.header('🔧 Model & App Settings')
model_choice = st.sidebar.selectbox('Choose Model', list(model_options.keys()), index=0)

# Model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', model_options[model_choice])
])

pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
preds = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

# Feature importance (for tree models)
def get_feature_importance(pipeline, features):
    reg = pipeline.named_steps['reg']
    if hasattr(reg, 'feature_importances_'):
        return reg.feature_importances_
    elif hasattr(reg, 'coef_'):
        return np.abs(reg.coef_)
    else:
        return np.zeros(len(features))

# Streamlit App
st.set_page_config(page_title="🌾 Crop Yield Predictor", layout="wide")
st.title("🌾 Crop Yield Predictor")
st.markdown("""
This app predicts crop yield based on agricultural and environmental features. 
Upload your data or use the form below to get instant predictions and insights for sustainable farming.
""")

with st.expander("See Model Performance"):
    st.write(f"**Model:** {model_choice}")
    st.write(f"**R² Score:** {score:.3f}")
    st.write(f"**RMSE:** {rmse:.3f}")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(y_test, preds, alpha=0.5, color='green')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Actual Yield')
    ax.set_ylabel('Predicted Yield')
    ax.set_title('Actual vs Predicted Yield')
    st.pyplot(fig)
    # Feature importance
    st.subheader('Feature Importance')
    importances = get_feature_importance(pipeline, features)
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    imp_df = imp_df.sort_values('Importance', ascending=False)
    st.bar_chart(imp_df.set_index('Feature'))

st.header("📋 Enter Crop Details for Prediction")

col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox('Crop', sorted(encoders['Crop'].classes_), help='Select the crop type')
    season = st.selectbox('Season', sorted(encoders['Season'].classes_), help='Select the season')
    state = st.selectbox('State', sorted(encoders['State'].classes_), help='Select the state')
    crop_year = st.number_input('Crop Year', min_value=1990, max_value=2025, value=2020, help='Year of cultivation')
with col2:
    area = st.number_input('Area (hectares)', min_value=1.0, value=1000.0, help='Area under cultivation')
    production = st.number_input('Production (tonnes)', min_value=1.0, value=1000.0, help='Total production')
    rainfall = st.number_input('Annual Rainfall (mm)', min_value=0.0, value=1000.0, help='Annual rainfall in mm')
    fertilizer = st.number_input('Fertilizer (kg)', min_value=0.0, value=1000.0, help='Fertilizer used in kg')
    pesticide = st.number_input('Pesticide (kg)', min_value=0.0, value=100.0, help='Pesticide used in kg')

if st.button('Predict Crop Yield'):
    # Encode categorical inputs
    crop_enc = encoders['Crop'].transform([crop])[0]
    season_enc = encoders['Season'].transform([season])[0]
    state_enc = encoders['State'].transform([state])[0]
    input_data = np.array([[crop_enc, crop_year, season_enc, state_enc, area, production, rainfall, fertilizer, pesticide]])
    pred_yield = pipeline.predict(input_data)[0]
    st.success(f"🌱 Predicted Crop Yield: **{pred_yield:.2f}**")
    st.info("This prediction can help optimize resource use and support sustainable agriculture.")
    # User feedback
    feedback = st.radio('Was this prediction helpful?', ['👍 Yes', '👎 No'], horizontal=True)
    if feedback == '👍 Yes':
        st.balloons()
        st.write('Thank you for your feedback!')
    else:
        st.write('Sorry! Please check your input values or try another model.')

st.markdown("---")
st.caption(f"Developed with ❤️ for sustainable agriculture. | Final Project | {pd.Timestamp.today().date()}")
