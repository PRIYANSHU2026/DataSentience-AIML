"""
Rainfall Prediction - Streamlit App
"""
import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="centered"
)

def load_model(model_path=None):
    """Load the pre-trained rainfall prediction model"""
    try:
        if model_path is None:
            # Use default model path relative to this file
            current_dir = Path(__file__).parent
            model_path = current_dir / "models" / "rf_model.pkl"
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
            
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_rainfall(input_data, model):
    """Make rainfall prediction using the loaded model"""
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of rain
        
        return "Rain" if prediction == 1 else "No Rain", probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def show_sample_data():
    """Display sample input data format"""
    st.subheader("Sample Input Format")
    sample_data = {
        'day': [15, 20, 25],
        'pressure': [1018.6, 1015.2, 1012.8],
        'maxtemp': [21.5, 23.1, 22.8],
        'temparature': [20.0, 21.5, 21.0],
        'mintemp': [19.2, 20.1, 19.8],
        'dewpoint': [18.5, 19.2, 18.9],
        'humidity': [88, 82, 85],
        'cloud': [85, 60, 75],
        'sunshine': [0.4, 0.7, 0.6],
        'winddirection': [70.0, 65.0, 80.0],
        'windspeed': [16.7, 14.2, 18.3]
    }
    st.dataframe(pd.DataFrame(sample_data))

def main():
    st.title("🌧️ Rainfall Prediction")
    st.markdown("""
    Predict whether it will rain tomorrow based on today's weather conditions.
    Enter the required weather parameters below or upload a CSV file with the data.
    """)
    
    # Initialize session state for model loading
    if 'model' not in st.session_state:
        with st.spinner("Loading prediction model..."):
            st.session_state.model = load_model()
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Manual Input", "File Upload"])
    
    with tab1:
        st.subheader("Enter Weather Data")
        
        # Create input fields for each parameter
        col1, col2 = st.columns(2)
        
        with col1:
            day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
            pressure = st.number_input("Atmospheric Pressure (hPa)", min_value=980.0, max_value=1050.0, value=1018.6, step=0.1)
            maxtemp = st.number_input("Maximum Temperature (°C)", min_value=-20.0, max_value=50.0, value=21.5, step=0.1)
            temperature = st.number_input("Average Temperature (°C)", min_value=-20.0, max_value=50.0, value=20.0, step=0.1)
            mintemp = st.number_input("Minimum Temperature (°C)", min_value=-20.0, max_value=50.0, value=19.2, step=0.1)
        
        with col2:
            dewpoint = st.number_input("Dew Point (°C)", min_value=-20.0, max_value=50.0, value=18.5, step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=88)
            cloud = st.number_input("Cloud Cover (%)", min_value=0, max_value=100, value=85)
            sunshine = st.number_input("Sunshine Hours", min_value=0.0, max_value=24.0, value=0.4, step=0.1)
            winddirection = st.number_input("Wind Direction (degrees)", min_value=0.0, max_value=360.0, value=70.0, step=1.0)
            windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=16.7, step=0.1)
        
        # Prepare input data
        input_data = {
            'day': day,
            'pressure': pressure,
            'maxtemp': maxtemp,
            'temparature': temperature,
            'mintemp': mintemp,
            'dewpoint': dewpoint,
            'humidity': humidity,
            'cloud': cloud,
            'sunshine': sunshine,
            'winddirection': winddirection,
            'windspeed': windspeed
        }
        
        # Make prediction
        if st.button("Predict Rainfall", key="predict_manual"):
            if st.session_state.model is not None:
                with st.spinner("Making prediction..."):
                    result, probability = predict_rainfall(input_data, st.session_state.model)
                    
                    if result is not None:
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Prediction Result")
                            if result == "Rain":
                                st.error(f"🌧️ **{result}** ({(probability*100):.1f}% chance)")
                            else:
                                st.success(f"☀️ **{result}** ({(probability*100):.1f}% chance)")
                        
                        with col2:
                            # Create a simple gauge chart
                            fig, ax = plt.subplots(figsize=(6, 1))
                            ax.barh([0], [probability*100], height=0.5, color='skyblue' if result == "Rain" else 'orange')
                            ax.set_xlim(0, 100)
                            ax.set_xticks([0, 50, 100])
                            ax.set_xticklabels(['0%', '50%', '100%'])
                            ax.set_yticks([])
                            ax.set_title("Rain Probability")
                            st.pyplot(fig, use_container_width=True)
                        
                        # Show recommendations
                        with st.expander("Recommendations"):
                            if result == "Rain":
                                st.warning("""
                                **Rain is likely tomorrow. Consider:**
                                - Carrying an umbrella or raincoat
                                - Planning indoor activities
                                - Checking for potential flooding in your area
                                """)
                            else:
                                st.info("""
                                **No rain expected tomorrow. Great for:**
                                - Outdoor activities
                                - Drying clothes outside
                                - Gardening or yard work
                                """)
    
    with tab2:
        st.subheader("Upload Weather Data")
        uploaded_file = st.file_uploader("Upload CSV file with weather data", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                
                # Show the uploaded data
                st.subheader("Uploaded Data")
                st.dataframe(df.head())
                
                # Check if all required columns are present
                required_columns = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 
                                  'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
                
                if all(col in df.columns for col in required_columns):
                    if st.button("Predict for All Rows", key="predict_file"):
                        if st.session_state.model is not None:
                            with st.spinner("Processing predictions..."):
                                # Make predictions for each row
                                predictions = []
                                probabilities = []
                                
                                for _, row in df.iterrows():
                                    input_data = row[required_columns].to_dict()
                                    result, prob = predict_rainfall(input_data, st.session_state.model)
                                    predictions.append(result)
                                    probabilities.append(prob)
                                
                                # Add predictions to the dataframe
                                df['prediction'] = predictions
                                df['rain_probability'] = [f"{p*100:.1f}%" for p in probabilities]
                                
                                # Show results
                                st.subheader("Prediction Results")
                                st.dataframe(df[['day', 'temparature', 'humidity', 'prediction', 'rain_probability']])
                                
                                # Show summary statistics
                                st.subheader("Summary Statistics")
                                rain_count = sum(1 for p in predictions if p == "Rain")
                                st.metric("Days with Rain Predicted", f"{rain_count} out of {len(predictions)}")
                                
                                # Show a simple bar chart of predictions
                                fig, ax = plt.subplots()
                                pd.Series(predictions).value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
                                ax.set_title("Rainfall Prediction Distribution")
                                ax.set_xlabel("Prediction")
                                ax.set_ylabel("Count")
                                st.pyplot(fig)
                else:
                    missing = [col for col in required_columns if col not in df.columns]
                    st.error(f"Missing required columns: {', '.join(missing)}")
                    st.info("Please ensure your CSV file contains all required columns.")
                    show_sample_data()
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            show_sample_data()

if __name__ == "__main__":
    main()
