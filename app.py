import streamlit as st
import pickle
import numpy as np

# Load model and scaler (if you used one)
model = pickle.load(open('model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))  # Uncomment if applicable

# Title
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient details below to predict the likelihood of diabetes.")

# Input fields
preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=80)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
st_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=25)

# Convert inputs into numpy array
input_data = np.array([[preg, glucose, bp, st_thickness, insulin, bmi, dpf, age]])

# If model trained on scaled data, apply scaler
# input_data = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ You might have diabetes. Please consult a doctor.")
    else:
        st.success("✅ You are healthy! No signs of diabetes detected.")
