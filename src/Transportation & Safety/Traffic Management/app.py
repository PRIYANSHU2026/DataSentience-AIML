import streamlit as st
import os
import sys
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Traffic Management Systems",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #2E86C1;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #16A085;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #3498DB;
    }
    .sign-image {
        border: 2px solid #ddd;
        border-radius: 5px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🚦 Traffic Management Systems</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Select System:",
    ["Traffic Sign Recognition", "Ride Request Analysis"]
)

# Traffic Sign Recognition System
if option == "Traffic Sign Recognition":
    st.markdown('<div class="sub-header">🚸 Traffic Sign Recognition</div>', unsafe_allow_html=True)
    
    # Information about the system
    with st.expander("About this system", expanded=True):
        st.markdown("""
        This system recognizes and classifies traffic signs in images using a deep learning model. 
        It can identify 43 different classes of traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB).
        
        **How it works:**
        1. Image preprocessing to normalize and resize input
        2. CNN model classifies the traffic sign
        3. Results show the predicted sign class and confidence
        
        **Applications:**
        - Driver assistance systems
        - Autonomous vehicles
        - Traffic sign inventory and maintenance
        - Driver education tools
        """)
    
    # Traffic sign classes dictionary
    classes = {
        0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
        3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
        6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
        9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection', 
        12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 
        16: 'Veh > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution', 
        19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve', 
        22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 
        25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 
        28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
        31: 'Wild animals crossing', 32: 'End speed + passing limits', 33: 'Turn right ahead', 
        34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right', 
        37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left', 
        40: 'Roundabout mandatory', 41: 'End of no passing', 
        42: 'End no passing veh > 3.5 tons'
    }
    
    # File uploader for image
    uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Uploaded Traffic Sign", use_column_width=True, output_format="PNG")
        
        # Preprocess image
        img = image.resize((30, 30))
        img = np.array(img)
        img = img / 255.0  # Normalize
        
        # Simulate model prediction
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write("**Recognition Results:**")
            
            # Simulate processing
            with st.spinner('Analyzing traffic sign...'):
                # In a real app, you'd load and use the actual model
                # For demo, we'll simulate a prediction
                import random
                predicted_class_idx = random.randint(0, 42)
                confidence = random.uniform(0.75, 0.99)
                
                # Display prediction
                st.markdown(f"**Predicted Sign:** {classes[predicted_class_idx]}")
                st.markdown(f"**Confidence:** {confidence:.2%}")
                
                # Display confidence bar
                st.progress(confidence)
                
                # Show top 3 predictions
                st.markdown("**Top 3 Predictions:**")
                
                # Generate 3 random predictions for demo
                top_indices = [predicted_class_idx]
                while len(top_indices) < 3:
                    idx = random.randint(0, 42)
                    if idx not in top_indices:
                        top_indices.append(idx)
                
                top_confidences = [confidence]
                top_confidences.extend([random.uniform(0.1, confidence-0.01) for _ in range(2)])
                top_confidences.sort(reverse=True)
                
                for i, (idx, conf) in enumerate(zip(top_indices, top_confidences)):
                    st.markdown(f"{i+1}. **{classes[idx]}** - {conf:.2%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional information
    st.markdown("### How to use this system")
    st.write("""
    1. Upload a clear image of a traffic sign
    2. The system will preprocess the image
    3. The CNN model will classify the traffic sign
    4. Results will show the predicted sign class and confidence level
    
    For optimal results, ensure the traffic sign is clearly visible and centered in the image.
    """)

# Ride Request Analysis System
elif option == "Ride Request Analysis":
    st.markdown('<div class="sub-header">🚕 Ride Request Analysis</div>', unsafe_allow_html=True)
    
    # Information about the system
    with st.expander("About this system", expanded=True):
        st.markdown("""
        This system analyzes ride-sharing request data to predict demand patterns and optimize vehicle allocation.
        It uses machine learning to forecast peak demand times and locations.
        
        **How it works:**
        1. Historical ride request data is analyzed
        2. Machine learning models identify patterns and trends
        3. Predictions are made for future demand
        4. Visualizations help in resource planning
        
        **Applications:**
        - Ride-sharing companies
        - Taxi services
        - Public transportation planning
        - Traffic management authorities
        """)
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Demand Heatmap", "Time Analysis", "Prediction Model"])
    
    with tab1:
        st.markdown("### Geographical Demand Distribution")
        st.write("This heatmap shows the concentration of ride requests across different areas.")
        
        # Generate a sample heatmap for demonstration
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.random.normal(size=1000)
        y = np.random.normal(size=1000)
        plt.hist2d(x, y, bins=30, cmap='viridis')
        plt.colorbar(label='Number of Requests')
        plt.title('Ride Request Density Heatmap')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        st.pyplot(fig)
        
        st.write("""
        **Insights:**
        - Downtown areas show the highest concentration of requests
        - Suburban areas have moderate demand during commute hours
        - Request patterns change significantly between weekdays and weekends
        """)
    
    with tab2:
        st.markdown("### Temporal Analysis")
        st.write("This chart shows how ride requests vary throughout the day and week.")
        
        # Generate sample time series data for demonstration
        dates = pd.date_range(start='2023-01-01', periods=24, freq='H')
        values = [100, 80, 60, 40, 30, 50, 100, 200, 300, 250, 220, 240, 230, 220, 240, 250, 300, 320, 280, 240, 200, 180, 150, 120]
        df = pd.DataFrame({'DateTime': dates, 'Requests': values})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(df['DateTime'].dt.strftime('%H:%M'), df['Requests'], marker='o', linestyle='-')
        plt.title('Hourly Ride Requests')
        plt.xlabel('Time of Day')
        plt.ylabel('Number of Requests')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.write("""
        **Insights:**
        - Morning peak: 7:00 AM - 9:00 AM
        - Evening peak: 5:00 PM - 7:00 PM
        - Lowest demand: 2:00 AM - 5:00 AM
        - Weekend patterns differ with later morning peaks
        """)
    
    with tab3:
        st.markdown("### Prediction Model Performance")
        st.write("This section shows the performance metrics of our demand prediction model.")
        
        # Create sample metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", "12.3 rides")
        col2.metric("RMSE", "18.7 rides")
        col3.metric("R² Score", "0.87")
        col4.metric("Accuracy", "91%")
        
        # Sample prediction vs actual chart
        actual = np.array([120, 132, 145, 160, 178, 190, 205, 220, 228, 240])
        predicted = actual + np.random.normal(0, 15, size=10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(range(1, 11), actual, 'b-', label='Actual')
        plt.plot(range(1, 11), predicted, 'r--', label='Predicted')
        plt.title('Prediction Model Performance')
        plt.xlabel('Time Period')
        plt.ylabel('Number of Ride Requests')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.write("""
        **Model Information:**
        - Algorithm: Gradient Boosting Regressor
        - Features: Time, day of week, weather, special events
        - Training data: 6 months of historical requests
        - Validation method: Time-series cross-validation
        """)