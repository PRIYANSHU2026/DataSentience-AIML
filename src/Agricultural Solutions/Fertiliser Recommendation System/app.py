"""
Fertilizer Recommendation System

AI-powered fertilizer recommendations based on soil conditions, crop type, and environmental factors.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path for shared components
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui_components import UIComponents

# Mock data for demonstration
CROP_TYPES = ["Rice", "Wheat", "Maize", "Cotton", "Soybean", "Sugarcane"]
SOIL_TYPES = ["Loamy", "Sandy", "Clay", "Silty", "Peaty", "Chalky"]
FERTILIZER_TYPES = ["Urea", "DAP", "MOP", "NPK", "SSP", "Ammonium Sulfate"]

# Mock prediction function
def get_fertilizer_recommendation(input_data):
    """Generate fertilizer recommendations based on input parameters"""
    # In a real application, this would use a trained ML model
    base_npk = {
        'Rice': (120, 60, 40), 'Wheat': (100, 50, 30), 
        'Maize': (150, 75, 60), 'Cotton': (80, 40, 50),
        'Soybean': (30, 80, 40), 'Sugarcane': (200, 100, 80)
    }
    
    crop = input_data['crop']
    base_n, base_p, base_k = base_npk.get(crop, (100, 50, 50))
    
    # Adjust based on soil conditions
    soil_factor = 1.0
    if input_data['soil_type'] == 'Sandy':
        soil_factor = 1.2  # Sandy soils need more frequent fertilization
    elif input_data['soil_type'] == 'Clay':
        soil_factor = 0.8  # Clay soils retain nutrients better
    
    # Adjust based on existing nutrient levels
    n_factor = max(0.5, 1.5 - (input_data['nitrogen'] / 100))
    p_factor = max(0.5, 1.5 - (input_data['phosphorus'] / 50))
    k_factor = max(0.5, 1.5 - (input_data['potassium'] / 100))
    
    # Calculate recommended amounts (kg/acre)
    n_rec = base_n * soil_factor * n_factor * (1 + input_data['growth_stage'] * 0.2)
    p_rec = base_p * soil_factor * p_factor
    k_rec = base_k * soil_factor * k_factor
    
    # Add some randomness for demonstration
    n_rec *= np.random.uniform(0.9, 1.1)
    p_rec *= np.random.uniform(0.9, 1.1)
    k_rec *= np.random.uniform(0.9, 1.1)
    
    return {
        'recommendations': {
            'Nitrogen (N)': round(n_rec, 1),
            'Phosphorus (P2O5)': round(p_rec, 1),
            'Potassium (K2O)': round(k_rec, 1)
        },
        'fertilizers': [
            {
                'name': 'Urea',
                'amount': round(n_rec * 2.17, 1),  # Urea is 46% N
                'timing': 'Split application: 50% at sowing, 25% at tillering, 25% at panicle initiation'
            },
            {
                'name': 'DAP',
                'amount': round(p_rec / 0.46, 1),  # DAP is 46% P2O5
                'timing': 'Basal application at sowing'
            },
            {
                'name': 'MOP',
                'amount': round(k_rec / 0.60, 1),  # MOP is 60% K2O
                'timing': 'Split application: 50% at sowing, 50% at active growth stage'
            }
        ]
    }

def display_recommendations(recommendation):
    """Display fertilizer recommendations with visualizations"""
    st.success("## 🌱 Fertilizer Recommendations")
    
    # Nutrient balance chart
    nutrients = list(recommendation['recommendations'].keys())
    values = list(recommendation['recommendations'].values())
    
    fig = px.bar(
        x=nutrients, 
        y=values,
        labels={'x': 'Nutrient', 'y': 'Amount (kg/acre)'},
        title="Recommended Nutrient Application Rates",
        color=nutrients,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed fertilizer recommendations
    st.markdown("### 📋 Recommended Fertilizers")
    
    for i, fert in enumerate(recommendation['fertilizers'], 1):
        with st.expander(f"{i}. {fert['name']}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Amount", f"{fert['amount']} kg/acre")
                st.metric("Timing", fert['timing'])
            with col2:
                # Add a placeholder for fertilizer image or additional info
                st.info(f"Application notes for {fert['name']} will be displayed here.")
    
    # Additional recommendations
    st.markdown("### 💡 Additional Recommendations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌧️ Application Tips")
        st.write("""
        - Apply fertilizers when soil moisture is adequate
        - Avoid application before heavy rainfall
        - Consider split applications for better efficiency
        - Adjust based on local soil test results
        """)
    
    with col2:
        st.markdown("#### ⚠️ Precautions")
        st.write("""
        - Wear protective gear when handling fertilizers
        - Store fertilizers in a cool, dry place
        - Keep away from children and animals
        - Follow local environmental regulations
        """)

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Fertilizer Recommendation - Agricultural Solutions",
        page_icon="🌿",
        layout="wide"
    )
    
    # Initialize session state
    if 'input_data' not in st.session_state:
        st.session_state.input_data = {
            'crop': 'Wheat',
            'soil_type': 'Loamy',
            'growth_stage': 1,  # 1: Early, 2: Mid, 3: Late
            'nitrogen': 50,
            'phosphorus': 30,
            'potassium': 60,
            'ph': 6.5,
            'area': 1.0
        }
    
    # Header
    UIComponents.header(
        "🌾 Fertilizer Recommendation System",
        "Get AI-powered fertilizer recommendations based on your soil and crop conditions"
    )
    
    # Sidebar form
    with st.sidebar:
        st.title("🌱 Input Parameters")
        
        with st.form("fertilizer_form"):
            st.subheader("Crop Information")
            crop = st.selectbox("Crop Type", CROP_TYPES, index=1)
            growth_stage = st.radio(
                "Growth Stage", 
                ["Early (0-25%)", "Middle (25-75%)", "Late (75-100%)"],
                index=1
            )
            
            st.subheader("Soil Analysis")
            soil_type = st.selectbox("Soil Type", SOIL_TYPES)
            
            col1, col2 = st.columns(2)
            with col1:
                nitrogen = st.slider("Nitrogen (kg/ha)", 0, 200, 50)
                phosphorus = st.slider("Phosphorus (kg/ha)", 0, 100, 30)
            with col2:
                potassium = st.slider("Potassium (kg/ha)", 0, 200, 60)
                ph = st.slider("pH Level", 4.0, 9.0, 6.5, 0.1)
            
            area = st.number_input("Area (acres)", 0.1, 1000.0, 1.0, 0.1)
            
            if st.form_submit_button("🌱 Get Recommendations"):
                st.session_state.input_data = {
                    'crop': crop,
                    'soil_type': soil_type,
                    'growth_stage': ["Early (0-25%)", "Middle (25-75%)", "Late (75-100%)"].index(growth_stage) + 1,
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'ph': ph,
                    'area': area
                }
                st.session_state.recommendation = get_fertilizer_recommendation(st.session_state.input_data)
    
    # Main content
    if 'recommendation' in st.session_state:
        display_recommendations(st.session_state.recommendation)
        
        # Export button
        st.download_button(
            "💾 Export Recommendations",
            pd.DataFrame({
                'Parameter': list(st.session_state.recommendation['recommendations'].keys()),
                'Amount (kg/acre)': list(st.session_state.recommendation['recommendations'].values())
            }).to_csv(index=False),
            "fertilizer_recommendations.csv",
            "text/csv"
        )
    else:
        st.info("👈 Enter your soil and crop parameters in the sidebar to get fertilizer recommendations.")
    
    # Back button
    if st.sidebar.button("⬅️ Back to Agricultural Solutions"):
        st.switch_page("../app.py")

if __name__ == "__main__":
    main()