"""
Pollution Spike Alert - Streamlit App
"""
import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(
    page_title="Pollution Spike Alert",
    page_icon="⚠️",
    layout="centered"
)

def load_model():
    """Load the pre-trained model and scaler"""
    try:
        # Get the directory of the current script
        current_dir = Path(__file__).parent
        
        # Load the model and scaler
        model_path = current_dir / "model" / "rf_spike_model.pkl"
        scaler_path = current_dir / "model" / "feature_scaler.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            st.error("Model files not found. Please make sure the model files exist in the 'model' directory.")
            return None, None
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_spike(data, model, scaler):
    """Make prediction using the loaded model"""
    try:
        # Ensure all required features are present
        required_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'Holidays_Count', 'Days', 'Month']
        
        # Check if all required features are in the data
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            st.error(f"Missing required features: {', '.join(missing_features)}")
            return None, None
            
        # Prepare the input data
        X = data[required_features].tail(1)  # Use the latest data point
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]  # Probability of spike (class 1)
        
        return pred, prob
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def plot_historical_data(data):
    """Plot historical AQI data"""
    try:
        if 'AQI' not in data.columns:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 4))
        data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
        data = data.sort_values('Date')
        
        # Plot AQI values
        ax.plot(data['Date'], data['AQI'], label='AQI', color='blue')
        
        # Mark spikes if available
        if 'spike' in data.columns:
            spikes = data[data['spike'] == 1]
            ax.scatter(spikes['Date'], spikes['AQI'], color='red', label='Spike Detected', zorder=5)
        
        ax.set_title('Historical AQI with Spike Alerts')
        ax.set_xlabel('Date')
        ax.set_ylabel('AQI')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"Could not generate historical plot: {str(e)}")
        return None

def main():
    st.title("⚠️ Pollution Spike Alert")
    st.markdown("""
    This tool predicts potential AQI (Air Quality Index) spikes based on current pollution levels.
    Upload a CSV file with the required pollution metrics to check for potential spikes.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload pollution data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)
            
            # Display the data
            st.subheader("Uploaded Data")
            st.dataframe(data.tail())  # Show last few rows
            
            # Load model and scaler
            model, scaler = load_model()
            
            if model is not None and scaler is not None:
                # Make prediction
                if st.button("Check for AQI Spike"):
                    with st.spinner("Analyzing data for potential spikes..."):
                        pred, prob = predict_spike(data, model, scaler)
                        
                        if pred is not None and prob is not None:
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Prediction Result")
                                if pred == 1:
                                    st.error(f"🚨 AQI Spike Predicted: **YES**")
                                else:
                                    st.success(f"✅ No AQI Spike Predicted")
                                
                                st.metric("Confidence", f"{prob*100:.1f}%")
                            
                            # Show historical plot if possible
                            historical_plot = plot_historical_data(data)
                            if historical_plot:
                                st.subheader("Historical AQI Data")
                                st.pyplot(historical_plot)
                            
                            # Show recommendations based on prediction
                            with st.expander("Recommendations"):
                                if pred == 1:
                                    st.warning("""
                                    **Recommended Actions:**
                                    - Issue public health advisory
                                    - Consider implementing traffic restrictions
                                    - Advise sensitive groups to stay indoors
                                    - Increase monitoring frequency
                                    """)
                                else:
                                    st.info("""
                                    **Status Normal**
                                    - Continue regular monitoring
                                    - Maintain current air quality measures
                                    """)
            
            # Show required data format
            with st.expander("Required Data Format"):
                st.markdown("""
                The CSV file should contain the following columns:
                - `PM2.5`: Particulate Matter 2.5 (µg/m³)
                - `PM10`: Particulate Matter 10 (µg/m³)
                - `NO2`: Nitrogen Dioxide (ppb)
                - `SO2`: Sulfur Dioxide (ppb)
                - `CO`: Carbon Monoxide (ppm)
                - `Ozone`: Ozone (ppb)
                - `Holidays_Count`: Number of holidays (0 or 1)
                - `Day`: Day of the month (1-31)
                - `Month`: Month (1-12)
                - `Year`: Year (e.g., 2023)
                - `AQI`: Air Quality Index (optional for prediction, needed for visualization)
                """)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Show sample data format
        st.info("""
        **Note:** Upload a CSV file with pollution data to check for potential AQI spikes.
        Use the "Required Data Format" section below to ensure your data is in the correct format.
        """)
        
        with st.expander("Sample Data Format"):
            sample_data = {
                'PM2.5': [45.2, 42.1, 47.8],
                'PM10': [78.5, 75.2, 82.3],
                'NO2': [28.4, 30.1, 32.7],
                'SO2': [12.5, 11.8, 13.2],
                'CO': [1.2, 1.3, 1.4],
                'Ozone': [45.6, 43.2, 48.1],
                'Holidays_Count': [0, 0, 1],
                'Day': [10, 11, 12],
                'Month': [8, 8, 8],
                'Year': [2023, 2023, 2023],
                'AQI': [156, 162, 178]
            }
            st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()
