import streamlit as st
import pickle
import numpy as np
import os

# ✅ Load the saved model safely
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

# ✅ App title
st.title("🩺 Diabetes Prediction using Machine Learning")

st.write("""
This app predicts whether a person is likely to have **diabetes**
based on medical attributes such as Glucose level, BMI, Insulin, etc.
""")

# ✅ Input fields
Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose Level", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0, format="%.1f")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
Age = st.number_input("Age", min_value=1, step=1)

# ✅ Prediction button
if st.button("🔍 Predict"):
    try:
        # Create a numpy array for the model input
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display result
        if prediction == 1:
            st.error("⚠️ The person is likely to have diabetes.")
        else:
            st.success("✅ The person is not likely to have diabetes.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ✅ Footer
st.markdown("---")
st.caption("Created by Hariom Dixit | ML Project")
