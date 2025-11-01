import streamlit as st
import pickle
import numpy as np

st.title("Diabetes Prediction App")

# Load model safely
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
age = st.number_input("Age", 0, 100)

if st.button("Predict"):
    features = np.array([[preg, glucose, bp, insulin, bmi, age]])
    result = model.predict(features)
    st.success("You are Diabetic" if result[0] == 1 else "You are Not Diabetic")
