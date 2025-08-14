"""
Lung Cancer Prediction - Streamlit App
"""
import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="centered"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1e88e5;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 500;
    }
    .safe {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }
    .risk {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

def load_model(model_path='models/rf_model.pkl'):
    """Load the trained model"""
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def prepare_input(data_dict):
    """Prepare input data for prediction"""
    data_dict['GENDER'] = 1 if data_dict['GENDER'] == 'M' else 0
    return pd.DataFrame([data_dict])

def predict_risk(input_data, model):
    """Make prediction using the model"""
    input_df = prepare_input(input_data)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # Probability of class 1 (At Risk)
    return prediction, probability

def show_prediction_result(prediction, probability):
    """Display prediction result"""
    if prediction == 1:
        st.markdown(
            f'<div class="result-box risk">'
            f'<h3>🚨 High Risk of Lung Cancer</h3>'
            f'<p>Probability: {probability*100:.1f}%</p>'
            f'<p>Please consult a healthcare professional for further evaluation.</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box safe">'
            f'<h3>✅ Low Risk of Lung Cancer</h3>'
            f'<p>Probability: {(1-probability)*100:.1f}%</p>'
            f'<p>Continue with healthy lifestyle choices and regular check-ups.</p>'
            f'</div>',
            unsafe_allow_html=True
        )

def main():
    """Main function to run the Streamlit app"""
    st.markdown('<h1 class="main-header">Lung Cancer Risk Assessment</h1>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        This tool helps assess the risk of lung cancer based on various health and lifestyle factors. 
        It uses a machine learning model trained on clinical data to provide a risk assessment.
        
        **Note:** This tool is for informational purposes only and should not be used as a substitute for 
        professional medical advice, diagnosis, or treatment.
        """)
    
    # Create form for user input
    with st.form("lung_cancer_form"):
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.radio("Gender", ["M", "F"], horizontal=True)
            age = st.slider("Age", 20, 100, 50)
            smoking = st.select_slider("Smoking (pack-years)", options=[0, 1], format_func=lambda x: "Non-smoker" if x == 0 else "Smoker")
            yellow_fingers = st.select_slider("Yellow Fingers", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            anxiety = st.select_slider("Anxiety", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            peer_pressure = st.select_slider("Peer Pressure", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
        with col2:
            chronic_disease = st.select_slider("Chronic Disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            fatigue = st.select_slider("Fatigue", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            allergy = st.select_slider("Allergy", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            wheezing = st.select_slider("Wheezing", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            alcohol_consuming = st.select_slider("Alcohol Consumption", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
        st.subheader("Symptoms")
        col3, col4 = st.columns(2)
        
        with col3:
            coughing = st.select_slider("Chronic Cough", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            shortness_of_breath = st.select_slider("Shortness of Breath", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
        with col4:
            swallowing_difficulty = st.select_slider("Difficulty Swallowing", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            chest_pain = st.select_slider("Chest Pain", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        # Submit button
        submitted = st.form_submit_button("🔍 Assess Risk")
        
        if submitted:
            # Create input dictionary
            input_data = {
                'GENDER': gender,
                'AGE': age,
                'SMOKING': smoking,
                'YELLOW_FINGERS': yellow_fingers,
                'ANXIETY': anxiety,
                'PEER_PRESSURE': peer_pressure,
                'CHRONIC_DISEASE': chronic_disease,
                'FATIGUE': fatigue,
                'ALLERGY': allergy,
                'WHEEZING': wheezing,
                'ALCOHOL_CONSUMING': alcohol_consuming,
                'COUGHING': coughing,
                'SHORTNESS_OF_BREATH': shortness_of_breath,
                'SWALLOWING_DIFFICULTY': swallowing_difficulty,
                'CHEST_PAIN': chest_pain
            }
            
            # Load model and make prediction
            try:
                model_path = os.path.join(os.path.dirname(__file__), 'models', 'rf_model.pkl')
                model = load_model(model_path)
                prediction, probability = predict_risk(input_data, model)
                
                # Show result
                show_prediction_result(prediction, probability)
                
                # Show prevention tips
                with st.expander("💡 Prevention Tips"):
                    st.markdown("""
                    - **Quit Smoking**: The most important step to reduce lung cancer risk
                    - **Avoid Secondhand Smoke**: Stay away from environments where people smoke
                    - **Test Your Home for Radon**: High radon levels can increase lung cancer risk
                    - **Eat a Healthy Diet**: Include plenty of fruits and vegetables
                    - **Exercise Regularly**: Aim for at least 30 minutes of moderate exercise most days
                    - **Limit Alcohol**: Excessive drinking can increase cancer risk
                    - **Protect Against Workplace Carcinogens**: Use protective equipment if exposed to harmful substances
                    """)
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    main()
