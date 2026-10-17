import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="❤️")

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.joblib")

model = load_model()

st.title("❤️ Heart Disease Risk Prediction")
st.caption("Educational decision-support demo. Not a medical diagnosis.")

st.subheader("Patient details")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Female", "Male"])
chest_pain_type = st.selectbox("Chest pain type", [0, 1, 2, 3])
resting_bp = st.number_input("Resting blood pressure", min_value=0, max_value=300, value=120)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=200)
fasting_blood_sugar = st.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
resting_ecg = st.selectbox("Resting ECG", [0, 1, 2])
max_heart_rate = st.number_input("Maximum heart rate", min_value=0, max_value=250, value=150)
exercise_angina = st.selectbox("Exercise angina", ["No", "Yes"])
oldpeak = st.number_input("Oldpeak", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox("ST slope", [0, 1, 2])

input_data = pd.DataFrame([{
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "chest pain type": chest_pain_type,
    "resting bp s": resting_bp,
    "cholesterol": cholesterol,
    "fasting blood sugar": 1 if fasting_blood_sugar == "Yes" else 0,
    "resting ecg": resting_ecg,
    "max heart rate": max_heart_rate,
    "exercise angina": 1 if exercise_angina == "Yes" else 0,
    "oldpeak": oldpeak,
    "ST slope": st_slope,
}])

if st.button("Predict heart disease risk"):
    try:
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0, 1])

        col1, col2 = st.columns(2)
        col1.metric("Heart disease probability", f"{probability:.1%}")
        col2.metric("Predicted class", "Heart disease" if prediction == 1 else "No heart disease")

        if prediction == 1:
            st.error("The model predicts heart disease at the default threshold.")
        else:
            st.success("The model does not predict heart disease at the default threshold.")

        st.info("This tool is for educational purposes only and should not replace professional medical advice.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
