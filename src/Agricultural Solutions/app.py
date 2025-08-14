"""
Agricultural Solutions Main Page - Streamlit UI
Comprehensive agricultural solutions dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from ui_components import UIComponents

def main():
    """Main Agricultural Solutions Dashboard"""
    
    # Page configuration
    st.set_page_config(
        page_title="Agricultural Solutions - DataSentience-AIML",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    UIComponents.header(
        "🌾 Agricultural Solutions",
        "AI-Powered Solutions for Modern Agriculture"
    )
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Agricultural modules
    modules = {
        "Air Quality Index": "Monitor air quality for crop planning",
        "Crop Recommendation System": "ML-based crop selection",
        "Crop Yield Prediction": "Predict yields using AI",
        "Fertiliser Recommendation System": "Optimize fertilizer usage",
        "Plant Disease Detection": "Identify plant diseases from leaf images.",
        "Plant Seedlings Classification": "Classify plant seedlings into 12 different species.",
        "Soil Classifier CNN": "Deep learning soil analysis",
        "Tomato Disease Detection": "Detect diseases from tomato leaf images"
    }
    
    selected_module = st.sidebar.selectbox(
        "Choose a Module:",
        list(modules.keys()),
        help="Select the agricultural solution you want to explore"
    )
    
    # Sidebar info
    with st.sidebar.expander("📊 About Agricultural Solutions"):
        st.write("""
        Our Agricultural Solutions leverage cutting-edge AI/ML technologies to:
        - Optimize crop selection and yield
        - Detect diseases early
        - Monitor environmental conditions
        - Provide data-driven recommendations
        """)
    
    # Main content based on selection
    if selected_module == "Air Quality Index":
        show_air_quality_dashboard()
    elif selected_module == "Crop Recommendation System":
        show_crop_recommendation_dashboard()
    elif selected_module == "Crop Yield Prediction":
        show_yield_prediction_dashboard()
    elif selected_module == "Fertiliser Recommendation System":
        show_fertilizer_recommendation_dashboard()
    elif selected_module == "Plant Disease Detection":
        show_plant_disease_detection_dashboard()
    elif selected_module == "Plant Seedlings Classification":
        show_seedlings_classification_dashboard()
    elif selected_module == "Soil Classifier CNN":
        show_soil_classifier_dashboard()
    elif selected_module == "Tomato Disease Detection":
        show_tomato_disease_detection_dashboard()
    else:
        show_module_placeholder(selected_module, modules[selected_module])
    
    # Footer
    UIComponents.footer()

def show_air_quality_dashboard():
    """Air Quality Index Dashboard"""
    st.markdown("### 🌬️ Air Quality Monitoring Dashboard")
    
    # Sample AQI data
    cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
    aqi_values = [156, 89, 67, 134, 178, 98]
    categories = ['Unhealthy', 'Moderate', 'Good', 'Unhealthy for Sensitive', 'Unhealthy', 'Moderate']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        UIComponents.metric_card("Current AQI", "134", "Unhealthy", "#FF6B6B")
    
    with col2:
        UIComponents.metric_card("PM2.5 Level", "89 μg/m³", "High", "#FFA726")
    
    with col3:
        UIComponents.metric_card("Recommendation", "Limit Outdoor Activity", "Immediate", "#EF5350")
    
    # AQI visualization
    st.markdown("#### 📊 City-wise AQI Comparison")
    
    df = pd.DataFrame({
        'City': cities,
        'AQI': aqi_values,
        'Category': categories
    })
    
    fig = px.bar(df, x='City', y='AQI', color='Category',
                 color_discrete_map={
                     'Good': '#4CAF50',
                     'Moderate': '#FFEB3B',
                     'Unhealthy for Sensitive': '#FF9800',
                     'Unhealthy': '#FF5722'
                 })
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Historical data
    st.markdown("#### 📈 Historical AQI Trends")
    
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    aqi_trend = np.random.normal(120, 30, 30)
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=dates,
        y=aqi_trend,
        mode='lines+markers',
        name='AQI Trend',
        line=dict(color='#2196F3', width=2)
    ))
    fig_trend.update_layout(
        title="30-Day AQI Trend",
        xaxis_title="Date",
        yaxis_title="AQI Value",
        height=300
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    with col2:
        if st.button("📊 Detailed Analysis"):
            st.switch_page("air quality index/app.py")
    
    with col3:
        if st.button("💾 Export Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="aqi_data.csv",
                mime="text/csv"
            )

def show_crop_recommendation_dashboard():
    """Crop Recommendation Dashboard"""
    # Redirect to the dedicated Crop Recommendation System app
    import streamlit as st
    from pathlib import Path
    import sys
    
    # Add the current directory to path to allow relative imports
    sys.path.append(str(Path(__file__).parent))
    
    try:
        # Import and run the dedicated app
        sys.path.append(str(Path(__file__).parent / "Crop-Recommendation-System-Using-Machine-Learning-main"))
        from app import main as crop_recommendation_app
        crop_recommendation_app()
    except Exception as e:
        st.error(f"Error loading the Crop Recommendation module: {str(e)}")
        st.info("Please ensure all required dependencies are installed and the module files exist.")
        
        if st.button("⬅️ Back to Main Menu"):
            # This will refresh the page and go back to the main menu
            st.switch_page("app.py")

def show_yield_prediction_dashboard():
    """Crop Yield Prediction Dashboard"""
    # Redirect to the dedicated Yield Prediction app
    import streamlit as st
    from pathlib import Path
    import sys
    
    # Add the current directory to path to allow relative imports
    sys.path.append(str(Path(__file__).parent))
    
    try:
        # Import and run the dedicated app
        sys.path.append(str(Path(__file__).parent / "Crop Yield Prediction"))
        from app import main as yield_prediction_app
        yield_prediction_app()
    except Exception as e:
        st.error(f"Error loading the Yield Prediction module: {str(e)}")
        st.info("Please ensure all required dependencies are installed and the module files exist.")
        
        if st.button("⬅️ Back to Main Menu"):
            # This will refresh the page and go back to the main menu
            st.switch_page("app.py")

def show_fertilizer_recommendation_dashboard():
    """Fertilizer Recommendation Dashboard"""
    # Redirect to the dedicated Fertilizer Recommendation app
    import streamlit as st
    from pathlib import Path
    import sys
    
    # Add the current directory to path to allow relative imports
    sys.path.append(str(Path(__file__).parent))
    
    try:
        # Import and run the dedicated app
        sys.path.append(str(Path(__file__).parent / "Fertiliser Recommendation System"))
        from app import main as fertilizer_recommendation_app
        fertilizer_recommendation_app()
    except Exception as e:
        st.error(f"Error loading the Fertilizer Recommendation module: {str(e)}")
        st.info("Please ensure all required dependencies are installed and the module files exist.")
        
        if st.button("⬅️ Back to Main Menu"):
            # This will refresh the page and go back to the main menu
            st.switch_page("app.py")

def show_plant_disease_detection_dashboard():
    """Plant Disease Detection Dashboard"""
    # Redirect to the dedicated Plant Disease Detection app
    import streamlit as st
    from pathlib import Path
    import sys
    
    # Add the current directory to path to allow relative imports
    sys.path.append(str(Path(__file__).parent))
    
    try:
        # Import and run the dedicated app
        sys.path.append(str(Path(__file__).parent / "Plant Disease Detection"))
        from app import main as plant_disease_detection_app
        plant_disease_detection_app()
    except Exception as e:
        st.error(f"Error loading the Plant Disease Detection module: {str(e)}")
        st.info("Please ensure all required dependencies are installed and the module files exist.")
        
        if st.button("⬅️ Back to Main Menu"):
            # This will refresh the page and go back to the main menu
            st.switch_page("app.py")

def show_module_placeholder(module_name: str, description: str):
    """Placeholder for modules not yet implemented"""
    st.markdown(f"### 🚧 {module_name}")
    st.write(description)
    
    # Show development status
    st.warning("This module is currently under development.")
    
    # Show development roadmap
    st.markdown("#### 🛠️ Development Roadmap")
    roadmap = [
        "Data collection and preprocessing",
        "Model training and validation",
        "UI/UX design and implementation",
        "Testing and optimization",
        "Deployment and monitoring"
    ]
    
    for i, item in enumerate(roadmap, 1):
        st.write(f"{i}. {item}")
    
    # Contact info
    st.markdown("#### 📞 Contact")
    st.write("For updates on this module, please check back later or contact the development team.")

if __name__ == "__main__":
    main()
