import streamlit as st
from predict import predict_risk

st.set_page_config(page_title="Lung Cancer Prediction", page_icon="🫁")
st.title("🫁 Lung Cancer Risk Prediction App")
st.markdown("Provide the following information to predict lung cancer risk.")

# Function to convert Yes/No to 1/0
def yn_to_binary(value):
    return 1 if value == "Yes" else 0

# Input form
with st.form("lung_cancer_form"):
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.slider("Age", 18, 100, 40)

    # Using Yes/No for better clarity
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
    anxiety = st.selectbox("Anxiety", ["No", "Yes"])
    peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
    chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    allergy = st.selectbox("Allergy", ["No", "Yes"])
    wheezing = st.selectbox("Wheezing", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"])
    coughing = st.selectbox("Coughing", ["No", "Yes"])
    short_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
    swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

    submitted = st.form_submit_button("Predict")

# On submit
if submitted:
    user_input = {
        'GENDER': gender,
        'AGE': age,
        'SMOKING': yn_to_binary(smoking),
        'YELLOW_FINGERS': yn_to_binary(yellow_fingers),
        'ANXIETY': yn_to_binary(anxiety),
        'PEER_PRESSURE': yn_to_binary(peer_pressure),
        'CHRONIC_DISEASE': yn_to_binary(chronic_disease),
        'FATIGUE': yn_to_binary(fatigue),
        'ALLERGY': yn_to_binary(allergy),
        'WHEEZING': yn_to_binary(wheezing),
        'ALCOHOL_CONSUMING': yn_to_binary(alcohol),
        'COUGHING': yn_to_binary(coughing),
        'SHORTNESS_OF_BREATH': yn_to_binary(short_breath),
        'SWALLOWING_DIFFICULTY': yn_to_binary(swallowing),
        'CHEST_PAIN': yn_to_binary(chest_pain)
    }

    # Predict
    result = predict_risk(user_input)

    st.markdown("---")
    st.success(f"🔍 Prediction: **{result}**")
