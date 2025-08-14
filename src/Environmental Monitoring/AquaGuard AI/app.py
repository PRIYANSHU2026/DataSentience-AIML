"""
AquaGuard AI - Water Quality Prediction System
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Set page configuration
st.set_page_config(
    page_title="AquaGuard AI - Water Quality Prediction",
    page_icon="💧",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .metric-box {
        background-color: #e6f2ff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1a75ff;
    }
    .prediction-card {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .safe { color: #009933; font-weight: bold; }
    .unsafe { color: #ff3300; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_DIR = Path(__file__).parent / "models"
DATA_FILE = Path(__file__).parent / "water_potability.csv"

# Water quality parameter descriptions and units
PARAM_INFO = {
    'ph': {
        'name': 'pH',
        'unit': '',
        'description': 'pH value of water (0-14)',
        'min': 0,
        'max': 14,
        'step': 0.1
    },
    'Hardness': {
        'name': 'Hardness',
        'unit': 'mg/L',
        'description': 'Capacity of water to precipitate soap',
        'min': 0,
        'max': 400,
        'step': 1
    },
    'Solids': {
        'name': 'Total Dissolved Solids',
        'unit': 'ppm',
        'description': 'Total dissolved solids in water',
        'min': 0,
        'max': 50000,
        'step': 100
    },
    'Chloramines': {
        'name': 'Chloramines',
        'unit': 'ppm',
        'description': 'Amount of Chloramines',
        'min': 0,
        'max': 15,
        'step': 0.1
    },
    'Sulfate': {
        'name': 'Sulfate',
        'unit': 'mg/L',
        'description': 'Amount of Sulfates dissolved in water',
        'min': 0,
        'max': 500,
        'step': 1
    },
    'Conductivity': {
        'name': 'Conductivity',
        'unit': 'μS/cm',
        'description': 'Electrical conductivity of water',
        'min': 0,
        'max': 1000,
        'step': 1
    },
    'Organic_carbon': {
        'name': 'Organic Carbon',
        'unit': 'ppm',
        'description': 'Amount of organic carbon in water',
        'min': 0,
        'max': 30,
        'step': 0.1
    },
    'Trihalomethanes': {
        'name': 'Trihalomethanes',
        'unit': 'μg/L',
        'description': 'Amount of Trihalomethanes',
        'min': 0,
        'max': 150,
        'step': 1
    },
    'Turbidity': {
        'name': 'Turbidity',
        'unit': 'NTU',
        'description': 'Cloudiness of water',
        'min': 0,
        'max': 10,
        'step': 0.1
    }
}

def load_sample_data():
    """Load sample data for demonstration"""
    try:
        if os.path.exists(DATA_FILE):
            return pd.read_csv(DATA_FILE)
        return None
    except Exception as e:
        st.warning(f"Error loading sample data: {str(e)}")
        return None

def predict_water_quality(input_data):
    """Predict water potability based on input parameters"""
    try:
        # In a real implementation, you would load and use the trained model here
        # For demonstration, we'll use a simple rule-based approach
        # Replace this with actual model prediction
        
        # This is a placeholder - in reality, you would use:
        # model = joblib.load(os.path.join(MODEL_DIR, 'water_quality_model.pkl'))
        # prediction = model.predict_proba([input_data])[0][1]
        
        # Simple rule-based prediction for demonstration
        # (This is just an example and not scientifically accurate)
        score = 0
        
        # pH (ideal range: 6.5-8.5)
        ph = input_data[0]
        if 6.5 <= ph <= 8.5:
            score += 0.2
            
        # TDS (ideal: < 500 ppm)
        tds = input_data[2]
        if tds < 500:
            score += 0.15
            
        # Chloramines (ideal: < 4 ppm)
        chloramines = input_data[3]
        if chloramines < 4:
            score += 0.15
            
        # Sulfate (ideal: < 250 mg/L)
        sulfate = input_data[4]
        if sulfate < 250:
            score += 0.15
            
        # Organic Carbon (ideal: < 10 ppm)
        organic_carbon = input_data[6]
        if organic_carbon < 10:
            score += 0.15
            
        # Trihalomethanes (ideal: < 80 μg/L)
        trihalomethanes = input_data[7]
        if trihalomethanes < 80:
            score += 0.2
            
        # Normalize to 0-1 range
        probability = min(max(score, 0), 1)
        
        return probability > 0.5, probability
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return False, 0.0

def display_water_quality_parameters(data):
    """Display interactive visualization of water quality parameters"""
    if data is None:
        return
        
    st.markdown("### Water Quality Parameters Distribution")
    
    # Select a parameter to visualize
    param = st.selectbox(
        "Select a parameter to visualize:",
        options=list(PARAM_INFO.keys()),
        format_func=lambda x: PARAM_INFO[x]['name']
    )
    
    # Create histogram
    fig = px.histogram(
        data, 
        x=param, 
        color='Potability',
        nbins=30,
        title=f"Distribution of {PARAM_INFO[param]['name']} by Potability",
        labels={param: f"{PARAM_INFO[param]['name']} ({PARAM_INFO[param]['unit']})"},
        color_discrete_map={0: '#ff4d4d', 1: '#4da6ff'}
    )
    
    # Add reference lines for ideal ranges (example values)
    if param == 'ph':
        fig.add_vline(x=6.5, line_dash="dash", line_color="green")
        fig.add_vline(x=8.5, line_dash="dash", line_color="green")
        fig.add_annotation(x=7.5, y=0.9, yref="paper", text="Ideal range: 6.5-8.5", showarrow=False)
    elif param == 'Hardness':
        fig.add_vline(x=60, line_dash="dash", line_color="green")
        fig.add_vline(x=120, line_dash="dash", line_color="green")
        fig.add_annotation(x=90, y=0.9, yref="paper", text="Ideal range: 60-120 mg/L", showarrow=False)
    # Add more reference lines for other parameters as needed
    
    st.plotly_chart(fig, use_container_width=True)

def display_prediction_result(is_potable, probability):
    """Display the prediction result with appropriate styling"""
    st.markdown("### Prediction Result")
    
    if is_potable:
        st.markdown(
            f"""
            <div class="prediction-card">
                <h2>✅ Water is <span class="safe">SAFE</span> to drink</h2>
                <p>Probability of being potable: <strong>{probability*100:.1f}%</strong></p>
                <p>This water sample meets the required quality standards for drinking water.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="prediction-card">
                <h2>❌ Water is <span class="unsafe">UNSAFE</span> to drink</h2>
                <p>Probability of being potable: <strong>{probability*100:.1f}%</strong></p>
                <p>This water sample does not meet the required quality standards for drinking water.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Display gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': 'Water Potability Score (%)'},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ff4d4d'},
                {'range': [50, 70], 'color': '#ffcc00'},
                {'range': [70, 100], 'color': '#4da6ff'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    st.title("💧 AquaGuard AI - Water Quality Prediction")
    st.markdown("""
        Welcome to AquaGuard AI, an intelligent water quality monitoring and prediction system. 
        This tool helps you assess the potability of water based on various quality parameters.
    """)
    
    # Load sample data
    sample_data = load_sample_data()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Water Quality Check", "Data Analysis", "About"])
    
    with tab1:
        st.markdown("### Enter Water Quality Parameters")
        
        # Create input form
        with st.form("water_quality_form"):
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            input_data = []
            param_order = list(PARAM_INFO.keys())
            
            # First column of parameters
            with col1:
                for param in param_order[:5]:
                    info = PARAM_INFO[param]
                    value = st.number_input(
                        f"{info['name']} ({info['unit']})",
                        min_value=info['min'],
                        max_value=info['max'],
                        value=(info['min'] + info['max']) / 2,
                        step=info['step'],
                        help=info['description']
                    )
                    input_data.append(value)
            
            # Second column of parameters
            with col2:
                for param in param_order[5:]:
                    info = PARAM_INFO[param]
                    value = st.number_input(
                        f"{info['name']} ({info['unit']})",
                        min_value=info['min'],
                        max_value=info['max'],
                        value=(info['min'] + info['max']) / 2,
                        step=info['step'],
                        help=info['description'],
                        key=param  # Add key to prevent duplicate key error
                    )
                    input_data.append(value)
            
            # Submit button
            submitted = st.form_submit_button("Check Water Quality")
            
            if submitted:
                # Make prediction
                is_potable, probability = predict_water_quality(input_data)
                st.session_state.prediction = (is_potable, probability)
                st.session_state.input_data = input_data
                
                # Scroll to results
                st.experimental_rerun()
        
        # Display prediction result if available
        if 'prediction' in st.session_state:
            is_potable, probability = st.session_state.prediction
            display_prediction_result(is_potable, probability)
            
            # Show parameter summary
            st.markdown("### Parameter Summary")
            param_values = {}
            for i, param in enumerate(PARAM_INFO.keys()):
                param_values[param] = st.session_state.input_data[i]
            
            st.dataframe(pd.DataFrame({
                'Parameter': [PARAM_INFO[p]['name'] for p in param_values.keys()],
                'Value': [f"{v} {PARAM_INFO[p]['unit']}" for p, v in param_values.items()],
                'Description': [PARAM_INFO[p]['description'] for p in param_values.keys()]
            }), hide_index=True)
    
    with tab2:
        if sample_data is not None:
            st.markdown("### Water Quality Dataset Analysis")
            
            # Show dataset summary
            st.dataframe(sample_data.describe())
            
            # Display parameter distribution
            display_water_quality_parameters(sample_data)
            
            # Correlation heatmap
            st.markdown("### Correlation Between Parameters")
            corr = sample_data.corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Sample data not available for analysis.")
    
    with tab3:
        st.markdown("""
        ## About AquaGuard AI
        
        AquaGuard AI is an intelligent water quality monitoring and prediction system that helps assess 
        the potability of water based on various quality parameters.
        
        ### How It Works
        
        1. Enter the water quality parameters in the "Water Quality Check" tab
        2. Click "Check Water Quality" to get an instant prediction
        3. View detailed analysis and visualizations in the "Data Analysis" tab
        
        ### Water Quality Parameters
        
        The system analyzes the following parameters to determine water potability:
        """)
        
        # Display parameter information in a table
        param_table = []
        for param, info in PARAM_INFO.items():
            param_table.append({
                'Parameter': info['name'],
                'Unit': info['unit'],
                'Description': info['description']
            })
        
        st.table(param_table)
        
        st.markdown("""
        ### Disclaimer
        
        This tool provides predictions based on machine learning models and should not be used as 
        the sole basis for making decisions about water safety. Always consult with water quality 
        professionals for critical applications.
        """)
    
    # Add footer
    st.markdown("---")
    st.markdown("© 2023 AquaGuard AI | Environmental Monitoring System")

if __name__ == "__main__":
    main()
