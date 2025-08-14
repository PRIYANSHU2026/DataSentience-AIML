import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        max-width: 850px;
        margin: auto;
        padding: 2rem;
        border-radius: 15px;
        background-color: #161b22;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.3);
    }
    h1, h2, h3 {
        text-align: center;
        font-weight: bold;
    }
    .stSelectbox, .stNumberInput {
        border-radius: 8px !important;
        padding: 5px !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50, #2e8b57);
        color: white;
        padding: 0.8em 2em;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2e8b57, #4CAF50);
        transform: scale(1.05);
    }
    .result-container {
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1.5rem;
    }
    .diabetic {
        background-color: rgba(255, 77, 77, 0.15);
        color: #ff4d4d;
        border: 2px solid #ff4d4d;
    }
    .non-diabetic {
        background-color: rgba(128, 255, 128, 0.15);
        color: #80ff80;
        border: 2px solid #80ff80;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/diabetes_rf_model.pkl")
    scaler = joblib.load("model/diabetes_scaler.pkl")
    le_gender = joblib.load("model/gender_encoder.pkl")
    le_smoking = joblib.load("model/smoking_encoder.pkl")
    return model, scaler, le_gender, le_smoking

model, scaler, le_gender, le_smoking = load_artifacts()

st.title("🩺 Diabetes Prediction App")
st.write("<p style='text-align: center;'>Fill in the details below to check the likelihood of diabetes.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=list(le_gender.classes_))
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    hypertension = st.selectbox("Hypertension", options=[0, 1], help="0 = No, 1 = Yes")
    heart_disease = st.selectbox("Heart Disease", options=[0, 1], help="0 = No, 1 = Yes")

with col2:
    smoking_history = st.selectbox("Smoking History", options=list(le_smoking.classes_))
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=30.0, value=5.8, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, max_value=500.0, value=120.0, step=1.0)

if st.button("🔍 Predict"):
    X = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

    X['gender'] = le_gender.transform(X['gender'])
    X['smoking_history'] = le_smoking.transform(X['smoking_history'])

    X_scaled = scaler.transform(X)


    pred = model.predict(X_scaled)[0]
    result = "🩸 **Diabetic**" if pred == 1 else "✅ **Non-Diabetic**"

    if pred == 1:
        st.markdown(f"<div class='result-container diabetic'>{result}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-container non-diabetic'>{result}</div>", unsafe_allow_html=True)
