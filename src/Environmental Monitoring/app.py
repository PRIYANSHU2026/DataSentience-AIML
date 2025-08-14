"""
Environmental Monitoring - Main Application
"""
import streamlit as st
import sys
from pathlib import Path
import os

# Initialize session state for tool selection
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = None

# Set page configuration
st.set_page_config(
    page_title="Environmental Monitoring",
    page_icon="🌍",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #f0f8f0 0%, #e0f0e0 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E8B57;
        font-weight: 500;
        margin: 1rem 0;
    }
    .tool-card {
        background-color: #f0f8f0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #2E8B57;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        height: 100%;
    }
    .tool-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .tool-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2E8B57;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .tool-description {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        color: #666;
        font-size: 0.9rem;
    }
    .back-button {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def load_tool(tool_name):
    """Load and run the selected tool"""
    try:
        # Add the tool directory to path
        tool_path = os.path.join(os.path.dirname(__file__), tool_name.replace(" ", " "))
        if tool_path not in sys.path:
            sys.path.append(tool_path)
        
        # Import and run the tool
        from app import main as tool_main
        tool_main()
        
    except Exception as e:
        st.error(f"Error loading {tool_name}: {str(e)}")
        st.info(f"Please make sure the {tool_name} module is properly set up.")

# Main header
st.title("🌍 Environmental Monitoring")
st.markdown("""
Welcome to the Environmental Monitoring dashboard. Select a tool from the options below to get started.
""")
st.markdown("---")
def go_back_to_main():
    st.session_state.selected_tool = None

# Display selected tool or main dashboard
if st.session_state.selected_tool:
    # Back button
    if st.button("← Back to Dashboard"):
        go_back_to_main()
    
    # AQI Predictor LSTM Tool
    if st.session_state.selected_tool == "aqi_predictor":
        st.markdown('<p class="sub-header">📊 AQI Predictor (LSTM)</p>', unsafe_allow_html=True)
        st.markdown("Predict Air Quality Index using LSTM model")
        
        try:
            # Import and run the AQI Predictor module
            sys.path.append(str(Path(__file__).parent / "AQI Predictor lstm"))
            from app import main as aqi_predictor_app
            aqi_predictor_app()
            
        except Exception as e:
            st.error(f"Error loading the AQI Predictor module: {str(e)}")
            st.info("Please ensure all required dependencies are installed and the module files exist.")
            
            if st.button("⬅️ Back to Dashboard"):
                go_back_to_main()

    # AquaGuard AI Tool
    elif st.session_state.selected_tool == "aquaguard_ai":
        st.markdown('<p class="sub-header">🌊 AquaGuard AI</p>', unsafe_allow_html=True)
        st.markdown("Water quality monitoring and prediction system")
        
        try:
            # Import and run the AquaGuard AI module
            sys.path.append(str(Path(__file__).parent / "AquaGuard AI"))
            from app import main as aquaguard_app
            aquaguard_app()
            
        except Exception as e:
            st.error(f"Error loading the AquaGuard AI module: {str(e)}")
            st.info("Please ensure all required dependencies are installed and the module files exist.")
            
            if st.button("⬅️ Back to Dashboard"):
                go_back_to_main()

else:
    # Main dashboard with tools
    st.markdown("### Available Tools")
    st.markdown("Select a tool from the options below to get started.")

    # Create columns for tool cards (3 columns for better layout with 5 tools)
    col1, col2, col3 = st.columns(3)

    with col1:
        # AQI Predictor Card
        with st.container():
            st.markdown("""
            <div style='border-radius: 10px; padding: 20px; background-color: #f0f8ff; margin-bottom: 20px;'>
                <h3>AQI Predictor</h3>
                <p>Predict Air Quality Index based on environmental parameters</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Open AQI Predictor", key="aqi_btn", use_container_width=True):
                st.session_state.selected_tool = "aqi_predictor"

    with col2:
        # AquaGuard AI Card
        with st.container():
            st.markdown("""
            <div style='border-radius: 10px; padding: 20px; background-color: #f0f8ff; margin-bottom: 20px;'>
                <h3>AquaGuard AI</h3>
                <p>Water quality monitoring and prediction system</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Open AquaGuard AI", key="aquaguard_btn", use_container_width=True):
                st.session_state.selected_tool = "aquaguard_ai"
    
    with col3:
        # Bird Species Classification Card
        with st.container():
            st.markdown("""
            <div style='border-radius: 10px; padding: 20px; background-color: #f0f8ff; margin-bottom: 20px;'>
                <h3>Bird Species Classifier</h3>
                <p>Identify bird species from images using deep learning</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Open Bird Classifier", key="bird_classifier_btn", use_container_width=True):
                st.session_state.selected_tool = "bird_classifier"
    
    with col4:
        # Fire Detection Card
        with st.container():
            st.markdown("""
            <div style='border-radius: 10px; padding: 20px; background-color: #f0f8ff; margin-bottom: 20px;'>
                <h3>Fire Detection</h3>
                <p>Detect fire in images using computer vision</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Open Fire Detection", key="fire_detection_btn", use_container_width=True):
                st.session_state.selected_tool = "fire_detection"
    
    # Second row of tool cards
    col5, col6, col7 = st.columns(3)
    
    with col7:
        # Waste Classification Card
        with st.container():
            st.markdown("""
            <div style='border-radius: 10px; padding: 20px; background-color: #f0f8ff; margin-bottom: 20px;'>
                <h3>Waste Classification</h3>
                <p>Classify waste as Organic or Recyclable</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Open Waste Classifier", key="waste_classifier_btn", use_container_width=True):
                st.session_state.selected_tool = "waste_classifier"
    
    with col5:
        # Pollution Spike Alert Card
        with st.container():
            st.markdown("""
            <div style='border-radius: 10px; padding: 20px; background-color: #f0f8ff; margin-bottom: 20px;'>
                <h3>Pollution Spike Alert</h3>
                <p>Predict AQI spikes from pollution data</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Open Pollution Alert", key="pollution_alert_btn", use_container_width=True):
                st.session_state.selected_tool = "pollution_alert"
    
    with col6:
        # Rainfall Prediction Card
        with st.container():
            st.markdown("""
            <div style='border-radius: 10px; padding: 20px; background-color: #f0f8ff; margin-bottom: 20px;'>
                <h3>Rainfall Prediction</h3>
                <p>Predict rainfall based on weather conditions</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Open Rainfall Prediction", key="rainfall_prediction_btn", use_container_width=True):
                st.session_state.selected_tool = "rainfall_prediction"

    # Tool content area
    if st.session_state.selected_tool:
        st.markdown("---")
        
        # Back button
        if st.button("← Back to Tools"):
            st.session_state.selected_tool = None
            st.experimental_rerun()
        
        # Load and show the selected tool
        if st.session_state.selected_tool == "aqi_predictor":
            try:
                # Add the AQI Predictor directory to path
                aqi_path = os.path.join(os.path.dirname(__file__), "AQI Predictor lstm")
                if aqi_path not in sys.path:
                    sys.path.append(aqi_path)
                
                # Import and run the AQI Predictor app
                try:
                    from app import main as aqi_main
                    aqi_main()
                except ImportError as e:
                    st.error(f"Failed to import AQI Predictor module: {str(e)}")
                except Exception as e:
                    st.error(f"Error loading AQI Predictor: {str(e)}")
                    st.info("Please make sure the AQI Predictor module is properly set up.")
                    
        elif st.session_state.selected_tool == "bird_classifier":
            try:
                # Add the Bird Species Classification directory to path
                bird_path = os.path.join(os.path.dirname(__file__), "Bird Species Classification Web App")
                if bird_path not in sys.path:
                    sys.path.append(bird_path)
                
                # Import and run the Bird Classifier app
                try:
                    from app import main as bird_main
                    bird_main()
                except ImportError as e:
                    st.error(f"Failed to import Bird Classifier module: {str(e)}")
                except Exception as e:
                    st.error(f"Error loading Bird Classifier: {str(e)}")
                    st.info("Please make sure the Bird Classifier module is properly set up.")
                    
        elif st.session_state.selected_tool == "fire_detection":
            try:
                # Add the Fire Detection directory to path
                fire_path = os.path.join(os.path.dirname(__file__), "Fire Detection")
                if fire_path not in sys.path:
                    sys.path.append(fire_path)
                
                # Import and run the Fire Detection app
                try:
                    from app import main as fire_main
                    fire_main()
                except ImportError as e:
                    st.error(f"Failed to import Fire Detection module: {str(e)}")
                except Exception as e:
                    st.error(f"Error loading Fire Detection: {str(e)}")
                    st.info("Please make sure the Fire Detection module is properly set up.")
                    
        elif st.session_state.selected_tool == "pollution_alert":
            try:
                # Add the Pollution-spike-alert directory to path
                pollution_path = os.path.join(os.path.dirname(__file__), "Pollution-spike-alert")
                if pollution_path not in sys.path:
                    sys.path.append(pollution_path)
                
                # Import and run the Pollution Alert app
                try:
                    from app import main as pollution_main
                    pollution_main()
                except ImportError as e:
                    st.error(f"Failed to import Pollution Alert module: {str(e)}")
                except Exception as e:
                    st.error(f"Error loading Pollution Alert: {str(e)}")
                    st.info("Please make sure the Pollution Alert module is properly set up.")
                    st.info(f"Python path: {sys.path}")
                    
        elif st.session_state.selected_tool == "rainfall_prediction":
            try:
                # Add the rainfall-prediction directory to path
                rainfall_path = os.path.join(os.path.dirname(__file__), "rainfall-prediction")
                if rainfall_path not in sys.path:
                    sys.path.append(rainfall_path)
                
                # Import and run the Rainfall Prediction app
                try:
                    from app import main as rainfall_main
                    rainfall_main()
                except ImportError as e:
                    st.error(f"Failed to import Rainfall Prediction module: {str(e)}")
                except Exception as e:
                    st.error(f"Error loading Rainfall Prediction: {str(e)}")
                    st.info("Please make sure the Rainfall Prediction module is properly set up.")
                    st.info(f"Python path: {sys.path}")
                    
        elif st.session_state.selected_tool == "waste_classifier":
            try:
                # Add the Waste Classification directory to path
                waste_path = os.path.join(os.path.dirname(__file__), "Waste Classification")
                if waste_path not in sys.path:
                    sys.path.append(waste_path)
                
                # Import and run the Waste Classification app
                try:
                    from app import main as waste_main
                    waste_main()
                except ImportError as e:
                    st.error(f"Failed to import Waste Classification module: {str(e)}")
                except Exception as e:
                    st.error(f"Error loading Waste Classification: {str(e)}")
                    st.info("Please make sure the Waste Classification module is properly set up.")
                    st.info(f"Python path: {sys.path}")
        
        elif st.session_state.selected_tool == "aquaguard_ai":
            if tool_card(
                "📊 AQI Predictor",
                "Predict Air Quality Index using LSTM model with historical data analysis.",
                "📊",
                "Open AQI Predictor",
                "aqi_btn"
            ):
                st.session_state.selected_tool = "aqi_predictor"
                st.experimental_rerun()
    
    with col2:
        # AquaGuard AI Card
        if tool_card(
            "💧 AquaGuard AI",
            "Water quality monitoring and prediction system using machine learning.",
            "💧",
            "Open AquaGuard AI",
            "aquaguard_btn"
        ):
            st.session_state.selected_tool = "aquaguard_ai"
            st.experimental_rerun()

# Add footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Environmental Monitoring System 2023 | Powered by DataSentience</p>
</div>
""", unsafe_allow_html=True)
