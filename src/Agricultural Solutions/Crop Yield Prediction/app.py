"""
Crop Yield Prediction System

AI-powered crop yield prediction based on agricultural and environmental parameters.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui_components import UIComponents

# Mock crop data for demonstration
CROP_TYPES = ["Wheat", "Rice", "Maize", "Soybean", "Cotton", "Sugarcane"]
STATES = ["Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Karnataka", "Andhra Pradesh"]
SEASONS = ["Kharif", "Rabi", "Zaid"]

# Mock prediction function
def predict_yield(input_data):
    """Mock function to predict crop yield"""
    # In a real application, this would use a trained ML model
    base_yield = {
        'Wheat': 3.5, 'Rice': 4.2, 'Maize': 3.8,
        'Soybean': 2.5, 'Cotton': 0.8, 'Sugarcane': 70.0
    }
    
    # Simple calculation for demonstration
    crop = input_data['crop']
    base = base_yield.get(crop, 1.0)
    
    # Adjust based on inputs (mock logic)
    temp_factor = 1 + (25 - input_data['temperature']) * 0.01
    rain_factor = min(1.5, input_data['rainfall'] / 1000)
    soil_factor = 0.5 + (input_data['nitrogen'] + input_data['phosphorus'] + input_data['potassium']) / 300
    
    predicted_yield = base * temp_factor * rain_factor * soil_factor
    
    # Add some randomness
    predicted_yield *= np.random.normal(1.0, 0.1)
    
    return round(predicted_yield, 2), {
        'factors': {
            'temperature': round(temp_factor, 2),
            'rainfall': round(rain_factor, 2),
            'soil_quality': round(soil_factor, 2)
        }
    }

def display_yield_analysis(prediction, factors):
    """Display yield prediction and analysis"""
    st.subheader("📊 Yield Prediction & Analysis")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Yield", f"{prediction} t/ha")
    with col2:
        st.metric("Yield Potential", "High" if prediction > 2.5 else "Medium" if prediction > 1.5 else "Low")
    with col3:
        st.metric("Confidence", f"{np.random.randint(75, 96)}%")
    
    # Factors affecting yield
    st.markdown("#### 📈 Factors Affecting Yield")
    factors_df = pd.DataFrame({
        'Factor': list(factors['factors'].keys()),
        'Impact': [f"{v}x" for v in factors['factors'].values()]
    })
    
    # Create a radar chart for factors
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(factors['factors'].values()),
        theta=list(factors['factors'].keys()),
        fill='toself',
        name='Impact Factors'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=30, b=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    # Page config
    st.set_page_config(
        page_title="Crop Yield Prediction - Agricultural Solutions",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = {
            'crop': 'Wheat',
            'state': 'Punjab',
            'season': 'Rabi',
            'area': 1.0,
            'temperature': 25.0,
            'rainfall': 500.0,
            'nitrogen': 100,
            'phosphorus': 50,
            'potassium': 50
        }
    
    # Header
    UIComponents.header(
        "🌾 Crop Yield Prediction",
        "Predict crop yields based on agricultural and environmental parameters"
    )
    
    # Sidebar form
    with st.sidebar:
        st.title("📋 Input Parameters")
        
        with st.form("yield_form"):
            st.subheader("Crop & Location")
            crop = st.selectbox("Select Crop", CROP_TYPES, index=0)
            state = st.selectbox("State/Region", STATES, index=0)
            season = st.selectbox("Growing Season", SEASONS, index=0)
            area = st.number_input("Area (hectares)", 0.1, 1000.0, 1.0, 0.1)
            
            st.subheader("Environmental Conditions")
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.5)
            with col2:
                rainfall = st.slider("Rainfall (mm)", 0.0, 2000.0, 500.0, 10.0)
            
            st.subheader("Soil Nutrients (kg/ha)")
            nitrogen = st.slider("Nitrogen (N)", 0, 200, 100, 5)
            phosphorus = st.slider("Phosphorus (P)", 0, 200, 50, 5)
            potassium = st.slider("Potassium (K)", 0, 200, 50, 5)
            
            if st.form_submit_button("🌱 Predict Yield"):
                input_data = {
                    'crop': crop,
                    'state': state,
                    'season': season,
                    'area': area,
                    'temperature': temperature,
                    'rainfall': rainfall,
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium
                }
                st.session_state.prediction_data = input_data
                st.session_state.prediction_result = predict_yield(input_data)
    
    # Main content
    if 'prediction_result' in st.session_state:
        prediction, factors = st.session_state.prediction_result
        input_data = st.session_state.prediction_data
        
        # Display results
        display_yield_analysis(prediction, factors)
        
        # Historical trends (mock data)
        st.markdown("#### 📅 Historical Yield Trends")
        years = list(range(2020, 2025))
        historical_data = {
            'Year': years,
            'Yield (t/ha)': [prediction * (0.8 + 0.1 * i) for i in range(len(years))]
        }
        
        fig = px.line(
            historical_data,
            x='Year',
            y='Yield (t/ha)',
            title=f"Historical Yield Trends for {input_data['crop']} in {input_data['state']}",
            markers=True
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("#### 💡 Recommendations")
        if factors['factors']['soil_quality'] < 0.9:
            st.warning("⚠️ Soil nutrient levels are suboptimal. Consider soil amendments.")
        if factors['factors']['rainfall'] < 0.8:
            st.warning("⚠️ Rainfall is below optimal levels. Consider irrigation.")
        if factors['factors']['temperature'] > 1.1:
            st.warning("⚠️ Temperature is above optimal range. Consider heat-resistant varieties.")
        
        # Export button
        st.download_button(
            "💾 Export Prediction Report",
            pd.DataFrame([{
                'Crop': input_data['crop'],
                'State': input_data['state'],
                'Season': input_data['season'],
                'Area (ha)': input_data['area'],
                'Predicted Yield (t/ha)': prediction,
                'Total Yield (t)': prediction * input_data['area']
            }]).to_csv(index=False),
            f"yield_prediction_{input_data['crop'].lower()}.csv",
            "text/csv"
        )
    else:
        st.info("👈 Enter your crop and field parameters in the sidebar to get yield predictions.")
    
    # Back button
    if st.sidebar.button("⬅️ Back to Agricultural Solutions"):
        st.switch_page("../app.py")

if __name__ == "__main__":
    main()