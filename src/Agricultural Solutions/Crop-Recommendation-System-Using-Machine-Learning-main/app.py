"""
Crop Recommendation System

AI-powered crop recommendations based on soil and environmental conditions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui_components import UIComponents

# Mock crop data
CROP_DATA = {
    'rice': {'name': 'Rice', 'season': 'Kharif/Rabi', 'water': 'High', 'ph': '5-7.5', 'temp': '20-35°C'},
    'wheat': {'name': 'Wheat', 'season': 'Rabi', 'water': 'Medium', 'ph': '6-7.5', 'temp': '10-25°C'},
    'maize': {'name': 'Maize', 'season': 'Kharif', 'water': 'Medium', 'ph': '5.5-7.5', 'temp': '18-32°C'},
    'cotton': {'name': 'Cotton', 'season': 'Kharif', 'water': 'High', 'ph': '5.5-8.5', 'temp': '21-38°C'}
}

# Page config
st.set_page_config(
    page_title="Crop Recommendation - Agricultural Solutions",
    page_icon="🌾",
    layout="wide"
)

def get_recommendations(soil_params):
    """Get crop recommendations based on soil parameters"""
    # Mock recommendation logic
    recommendations = [
        {'crop': 'Rice', 'score': 0.92, 'suitability': 'Excellent'},
        {'crop': 'Wheat', 'score': 0.85, 'suitability': 'Very Good'},
        {'crop': 'Maize', 'score': 0.78, 'suitability': 'Good'}
    ]
    return recommendations

def display_crop_card(crop, index):
    """Display crop information card"""
    details = CROP_DATA.get(crop['crop'].lower(), {})
    with st.expander(f"{index + 1}. {crop['crop']} - {crop['suitability']}"):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(f"https://source.unsplash.com/200x150/?{crop['crop'].lower()},agriculture", 
                    use_column_width=True)
            st.progress(crop['score'])
            st.caption(f"Score: {crop['score']*100:.1f}%")
        with col2:
            if details:
                st.markdown(f"**Season:** {details.get('season', 'N/A')}")
                st.markdown(f"**Water Needs:** {details.get('water', 'N/A')}")
                st.markdown(f"**pH Range:** {details.get('ph', 'N/A')}")
                st.markdown(f"**Temperature:** {details.get('temp', 'N/A')}")

def main():
    """Main application function"""
    # Initialize session state
    if 'soil_params' not in st.session_state:
        st.session_state.soil_params = {
            'nitrogen': 90, 'phosphorus': 42, 'potassium': 43,
            'temperature': 25.0, 'humidity': 70.0, 'ph': 6.5, 'rainfall': 200.0
        }
    
    # Header
    UIComponents.header(
        "🌱 Crop Recommendation System",
        "Get AI-powered crop recommendations based on your soil and climate conditions"
    )
    
    # Sidebar form
    with st.sidebar:
        st.title("🌡️ Input Parameters")
        
        with st.form("soil_form"):
            st.subheader("Soil Nutrients (ppm)")
            col1, col2 = st.columns(2)
            with col1:
                nitrogen = st.slider("Nitrogen (N)", 0, 140, 90)
                phosphorus = st.slider("Phosphorus (P)", 5, 145, 42)
            with col2:
                potassium = st.slider("Potassium (K)", 5, 205, 43)
                ph = st.slider("pH Level", 3.0, 10.0, 6.5, 0.1)
            
            st.subheader("Climate")
            temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.5)
            humidity = st.slider("Humidity (%)", 0, 100, 70)
            rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 200.0, 5.0)
            
            if st.form_submit_button("🌾 Get Recommendations"):
                st.session_state.soil_params = {
                    'nitrogen': nitrogen, 'phosphorus': phosphorus, 'potassium': potassium,
                    'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall
                }
                st.session_state.recommendations = get_recommendations(st.session_state.soil_params)
    
    # Main content
    if 'recommendations' in st.session_state:
        st.success("## 🌟 Recommended Crops")
        for i, crop in enumerate(st.session_state.recommendations):
            display_crop_card(crop, i)
        
        st.download_button(
            "💾 Export Recommendations",
            pd.DataFrame(st.session_state.recommendations).to_csv(index=False),
            "crop_recommendations.csv",
            "text/csv"
        )
    else:
        st.info("👈 Enter your soil and climate parameters in the sidebar to get recommendations.")
    
    if st.sidebar.button("⬅️ Back to Agricultural Solutions"):
        st.switch_page("../app.py")

if __name__ == "__main__":
    main()
