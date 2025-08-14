import streamlit as st
import joblib
import numpy as np

MODEL_PATH = "model/heart_disease_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.markdown("""
    <style>
    .card {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 0px 15px rgba(255, 75, 75, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#ff4b4b;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Fill in the details below to check your risk</p>", unsafe_allow_html=True)

with st.form("heart_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.subheader("👤 Personal & Vital Stats")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=20, max_value=100, value=30)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    with c2:
        trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    st.subheader("🏃 ECG & Exercise Results")
    c3, c4 = st.columns(2)
    with c3:
        restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
    with c4:
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST (0-2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

    st.subheader("🧬 Thalassemia Test")
    thal = st.selectbox("Thal (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])
    
    submitted = st.form_submit_button("🔍 Predict Risk")
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(features)[0]

        try:
            proba = model.predict_proba(features)[0][prediction]
        except AttributeError:
            proba = None  


        if prediction == 1:
            st.error(f"💔 **High risk of Heart Disease!**" +
                     (f" _(Confidence: {proba*100:.2f}%)_" if proba is not None else ""))
        else:
            st.success(f"💖 **Low risk of Heart Disease!**" +
                       (f" _(Confidence: {proba*100:.2f}%)_" if proba is not None else ""))

        st.markdown("Stay healthy! 🫀")
