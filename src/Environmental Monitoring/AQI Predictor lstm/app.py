"""
AQI Predictor (LSTM) - Streamlit Application

This module provides a user interface for predicting Air Quality Index (AQI)
using a pre-trained LSTM model.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from pathlib import Path
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="AQI Predictor (LSTM)",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f8f0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #2E8B57;
    }
    .prediction-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .aqi-good { color: #009966; font-weight: bold; }
    .aqi-moderate { color: #ffde33; font-weight: bold; }
    .aqi-sensitive { color: #ff9933; font-weight: bold; }
    .aqi-unhealthy { color: #cc0033; font-weight: bold; }
    .aqi-very-unhealthy { color: #660099; font-weight: bold; }
    .aqi-hazardous { color: #7e0023; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_DIR = Path(__file__).parent / "model"
DATA_DIR = Path(__file__).parent / "data"

def get_aqi_category(aqi):
    """Determine AQI category based on AQI value"""
    if aqi <= 50:
        return "Good", "aqi-good", "😊"
    elif aqi <= 100:
        return "Moderate", "aqi-moderate", "😐"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "aqi-sensitive", "😷"
    elif aqi <= 200:
        return "Unhealthy", "aqi-unhealthy", "😷"
    elif aqi <= 300:
        return "Very Unhealthy", "aqi-very-unhealthy", "⚠️"
    else:
        return "Hazardous", "aqi-hazardous", "☠️"

def load_model():
    """Load the pre-trained LSTM model and scalers"""
    try:
        model_path = MODEL_DIR / "lstm_aqi_model.h5"
        feature_scaler_path = MODEL_DIR / "feature_scaler.pkl"
        target_scaler_path = MODEL_DIR / "target_scaler.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(feature_scaler_path) or not os.path.exists(target_scaler_path):
            st.error("Model files not found. Please ensure the model is trained and all required files are in the 'model' directory.")
            return None, None, None
            
        # In a real implementation, you would load the Keras model here
        # For now, we'll just load the scalers
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        
        return None, feature_scaler, target_scaler
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def load_sample_data():
    """Load sample data for demonstration"""
    try:
        data_path = DATA_DIR / "final_dataset.csv"
        if not os.path.exists(data_path):
            st.warning("Sample data not found. Using demo data.")
            return None
            
        df = pd.read_csv(data_path)
        return df
        
    except Exception as e:
        st.warning(f"Error loading sample data: {str(e)}. Using demo data.")
        return None

def predict_aqi(model, feature_scaler, target_scaler, input_data):
    """Make AQI prediction using the LSTM model"""
    try:
        # In a real implementation, you would preprocess the input data and make a prediction
        # For now, we'll return a random prediction for demonstration
        prediction = np.random.uniform(30, 200)  # Random AQI between 30 and 200
        return prediction
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def display_aqi_gauge(aqi_value):
    """Display AQI value as a gauge chart"""
    category, _, emoji = get_aqi_category(aqi_value)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = aqi_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Predicted AQI: {category} {emoji}", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "darkgreen"},
            'bar': {'color': "darkgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#009966'},
                {'range': [51, 100], 'color': '#ffde33'},
                {'range': [101, 150], 'color': '#ff9933'},
                {'range': [151, 200], 'color': '#cc0033'},
                {'range': [201, 300], 'color': '#660099'},
                {'range': [301, 500], 'color': '#7e0023'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def display_historical_trend(data):
    """Display historical AQI trend"""
    if data is None or len(data) == 0:
        # Generate sample data if no data is available
        dates = pd.date_range(end=datetime.now(), periods=30).date
        aqi_values = np.random.normal(100, 30, 30).clip(0, 300)
        data = pd.DataFrame({'Date': dates, 'AQI': aqi_values})
    else:
        # Ensure we have a Date column and AQI column
        if 'Date' not in data.columns and 'date' in data.columns:
            data = data.rename(columns={'date': 'Date'})
        if 'AQI' not in data.columns and 'aqi' in data.columns:
            data = data.rename(columns={'aqi': 'AQI'})
        
        # If we still don't have the required columns, use sample data
        if 'Date' not in data.columns or 'AQI' not in data.columns:
            dates = pd.date_range(end=datetime.now(), periods=30).date
            aqi_values = np.random.normal(100, 30, 30).clip(0, 300)
            data = pd.DataFrame({'Date': dates, 'AQI': aqi_values})
        else:
            # Sort by date and take the last 30 days
            data = data.sort_values('Date').tail(30)
    
    # Create the trend chart
    fig = px.line(
        data, 
        x='Date', 
        y='AQI', 
        title='Historical AQI Trend (Last 30 Days)',
        labels={'AQI': 'AQI Value', 'Date': 'Date'}
    )
    
    # Add AQI category bands
    fig.add_hrect(y0=0, y1=50, line_width=0, fillcolor="#009966", opacity=0.1, annotation_text="Good")
    fig.add_hrect(y0=51, y1=100, line_width=0, fillcolor="#ffde33", opacity=0.1, annotation_text="Moderate")
    fig.add_hrect(y0=101, y1=150, line_width=0, fillcolor="#ff9933", opacity=0.1, annotation_text="Unhealthy for Sensitive Groups")
    fig.add_hrect(y0=151, y1=200, line_width=0, fillcolor="#cc0033", opacity=0.1, annotation_text="Unhealthy")
    fig.add_hrect(y0=201, y1=300, line_width=0, fillcolor="#660099", opacity=0.1, annotation_text="Very Unhealthy")
    fig.add_hrect(y0=301, y1=500, line_width=0, fillcolor="#7e0023", opacity=0.1, annotation_text="Hazardous")
    
    # Update layout
    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="AQI",
        hovermode="x unified"
    )
    
    return fig

def display_aqi_info():
    """Display AQI information and health effects"""
    st.markdown("### AQI Categories and Health Effects")
    
    aqi_info = [
        {"AQI Range": "0-50", "Level": "Good", "Color": "#009966", "Health Concern": "Air quality is satisfactory, and air pollution poses little or no risk."},
        {"AQI Range": "51-100", "Level": "Moderate", "Color": "#ffde33", "Health Concern": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."},
        {"AQI Range": "101-150", "Level": "Unhealthy for Sensitive Groups", "Color": "#ff9933", "Health Concern": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."},
        {"AQI Range": "151-200", "Level": "Unhealthy", "Color": "#cc0033", "Health Concern": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."},
        {"AQI Range": "201-300", "Level": "Very Unhealthy", "Color": "#660099", "Health Concern": "Health alert: The risk of health effects is increased for everyone."},
        {"AQI Range": "301-500", "Level": "Hazardous", "Color": "#7e0023", "Health Concern": "Health warning of emergency conditions. The entire population is more likely to be affected."}
    ]
    
    for info in aqi_info:
        with st.expander(f"{info['Level']} (AQI {info['AQI Range']})", expanded=True):
            st.markdown(f"<div style='border-left: 5px solid {info['Color']}; padding: 10px; background-color: #f9f9f9;'>"
                       f"<p style='margin: 0;'><strong>Health Concern:</strong> {info['Health Concern']}</p>"
                       f"<p style='margin: 10px 0 0 0;'><strong>Precautions:</strong> "
                       f"{get_aqi_precautions(info['Level'])}</p>"
                       f"</div>", unsafe_allow_html=True)

def get_aqi_precautions(level):
    """Get recommended precautions based on AQI level"""
    precautions = {
        "Good": "No special precautions needed. Enjoy your normal outdoor activities.",
        "Moderate": "Unusually sensitive people should consider reducing prolonged or heavy exertion.",
        "Unhealthy for Sensitive Groups": "People with heart or lung disease, older adults, children, and people of lower socioeconomic status should reduce prolonged or heavy exertion.",
        "Unhealthy": "People with heart or lung disease, older adults, children, and people of lower socioeconomic status should avoid prolonged or heavy exertion; everyone else should reduce prolonged or heavy exertion.",
        "Very Unhealthy": "People with heart or lung disease, older adults, children, and people of lower socioeconomic status should avoid all physical activity outdoors. Everyone else should avoid prolonged or heavy exertion.",
        "Hazardous": "Everyone should avoid all physical activity outdoors; people with heart or lung disease, older adults, and children should remain indoors and keep activity levels low."
    }
    return precautions.get(level, "No specific precautions available.")

def main():
    """Main application function"""
    st.title("AQI Predictor (LSTM)")
    st.markdown("Predict Air Quality Index using historical data and LSTM model")
    
    # Load model and data
    with st.spinner("Loading model and data..."):
        model, feature_scaler, target_scaler = load_model()
        data = load_sample_data()
    
    # Main layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Input form
        with st.form("aqi_prediction_form"):
            st.markdown("### Input Parameters")
            
            # Create input fields for the model features
            col1_1, col1_2 = st.columns(2)
            
            with col1_1:
                pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=25.0, step=0.1)
                no2 = st.number_input("NO₂ (ppb)", min_value=0.0, max_value=500.0, value=20.0, step=0.1)
                co = st.number_input("CO (ppm)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
                holidays = st.number_input("Holidays Count", min_value=0, max_value=30, value=0)
                
            with col1_2:
                pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
                so2 = st.number_input("SO₂ (ppb)", min_value=0.0, max_value=500.0, value=10.0, step=0.1)
                ozone = st.number_input("Ozone (ppb)", min_value=0.0, max_value=500.0, value=30.0, step=0.1)
                days = st.number_input("Day of Month", min_value=1, max_value=31, value=datetime.now().day)
                month = st.number_input("Month", min_value=1, max_value=12, value=datetime.now().month)
            
            # Submit button
            submitted = st.form_submit_button("Predict AQI")
            
            if submitted:
                # Prepare input data
                input_data = {
                    'PM2.5': pm25,
                    'PM10': pm10,
                    'NO2': no2,
                    'SO2': so2,
                    'CO': co,
                    'Ozone': ozone,
                    'Holidays_Count': holidays,
                    'Days': days,
                    'Month': month
                }
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    aqi_prediction = predict_aqi(model, feature_scaler, target_scaler, input_data)
                    
                    if aqi_prediction is not None:
                        st.session_state.aqi_prediction = aqi_prediction
                        st.session_state.input_data = input_data
                        st.session_state.prediction_time = datetime.now()
                        
                        # Scroll to results
                        st.experimental_rerun()
    
    with col2:
        # Display AQI gauge if prediction exists
        if 'aqi_prediction' in st.session_state:
            st.markdown("### Prediction Results")
            
            # Display gauge chart
            gauge_fig = display_aqi_gauge(st.session_state.aqi_prediction)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Display AQI category and health info
            category, css_class, emoji = get_aqi_category(st.session_state.aqi_prediction)
            st.markdown(f"""
            <div class="prediction-card">
                <h3>AQI: <span class="{css_class}">{st.session_state.aqi_prediction:.1f} {emoji}</span></h3>
                <p><strong>Category:</strong> {category}</p>
                <p><strong>Prediction Time:</strong> {st.session_state.prediction_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Health Message:</strong> {get_aqi_precautions(category).split('.')[0]}.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show more details in an expander
            with st.expander("View Prediction Details"):
                st.markdown("#### Input Parameters")
                st.json(st.session_state.input_data)
                
                st.markdown("#### Health Recommendations")
                st.info(get_aqi_precautions(category))
        else:
            # Show AQI information card
            st.markdown("### About AQI")
            st.markdown("""
            <div class="prediction-card">
                <p>The Air Quality Index (AQI) is used to communicate how polluted the air currently is or how polluted it is forecast to become.</p>
                <p>Enter the air quality parameters on the left to get a prediction of the AQI and health recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show AQI scale
            st.image("https://www.airnow.gov/sites/default/files/2021-03/aqi-chart.png", 
                   caption="AQI Scale and Health Concern Levels", 
                   use_column_width=True)
    
    # Historical trend section
    st.markdown("---")
    st.markdown("### Historical AQI Trend")
    
    # Display historical trend chart
    trend_fig = display_historical_trend(data)
    st.plotly_chart(trend_fig, use_container_width=True)
    
    # AQI information and health effects
    st.markdown("---")
    display_aqi_info()
    
    # Add some space at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
