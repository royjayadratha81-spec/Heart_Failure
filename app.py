import streamlit as st
import joblib as jb
import numpy as np 
# Load the trained model and scaler
model = jb.load('heart_disease_model.pkl')
scaler = jb.load('scaler.pkl')
st.title("Heart Disease Prediction")
st.image("C:\\Users\\ROY\\Downloads\\Understanding-How-Heart-Disease-Impacts-Your-Body.webp", use_column_width=True)
caption = "Human Heart Anatomy"
st.caption(caption) 
width = st.slider("Select the width of the image", 100, 800, 400)
st.write("Please enter the following details to predict the presence of heart disease:")
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex= st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (in mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", options=[1, 2, 3])
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    result = "Presence of Heart Disease" if prediction[0] == 1 else "No Heart Disease"
    st.write("Prediction:", result) 