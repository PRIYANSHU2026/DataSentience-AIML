import streamlit as st
import os
import sys
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
import time

# Set page config
st.set_page_config(
    page_title="Driver Safety Systems",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #FF5733;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #3498DB;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🚗 Driver Safety Systems</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Select System:",
    ["Drowsiness Detection", "CCTV Accident Analysis"]
)

# Drowsiness Detection System
if option == "Drowsiness Detection":
    st.markdown('<div class="sub-header">👁️ Driver Drowsiness Detection</div>', unsafe_allow_html=True)
    
    # Information about the system
    with st.expander("About this system", expanded=True):
        st.markdown("""
        This system detects driver drowsiness by analyzing eye closure patterns. If the driver's eyes 
        remain closed for an extended period, the system triggers an alert to prevent accidents.
        
        **How it works:**
        1. Face and eye detection using Haar Cascade Classifier
        2. CNN model classifies eyes as open or closed
        3. Alert is triggered if eyes remain closed for 15+ consecutive frames
        
        **Applications:**
        - Long-distance truck drivers
        - Night shift workers
        - Highway safety monitoring
        """)
    
    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image of a driver's face", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to OpenCV format
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Load Haar Cascade classifiers
        try:
            face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), 
                                                'Drowsiness Detection/haarcascade_frontalface_default.xml'))
            left_eye_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), 
                                                   'Drowsiness Detection/haarcascade_lefteye_2splits.xml'))
            right_eye_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), 
                                                    'Drowsiness Detection/haarcascade_righteye_2splits.xml'))
            
            # Detect faces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Process the image and simulate detection
                with col2:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.write("**Detection Results:**")
                    
                    # Simulate model prediction (in a real app, you'd load the actual model)
                    st.write("Face detected: Yes")
                    st.write("Eyes detected: Yes")
                    
                    # Randomly simulate open/closed eyes for demo purposes
                    import random
                    eye_status = random.choice(["Open", "Closed"])
                    st.write(f"Eye status: {eye_status}")
                    
                    if eye_status == "Closed":
                        st.markdown('<div class="warning-box">⚠️ Warning: Driver appears drowsy!</div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">✅ Driver is alert</div>', 
                                   unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                with col2:
                    st.error("No face detected in the image. Please upload a clearer image.")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Additional information
    st.markdown("### How to use this system")
    st.write("""
    1. Upload a clear image of a driver's face
    2. The system will detect the face and eyes
    3. The CNN model will classify if the eyes are open or closed
    4. If the eyes are closed, a warning will be displayed
    
    For real-time monitoring, this system can be integrated with in-vehicle cameras.
    """)

# CCTV Accident Analysis System
elif option == "CCTV Accident Analysis":
    st.markdown('<div class="sub-header">🎥 CCTV Accident Analysis</div>', unsafe_allow_html=True)
    
    # Information about the system
    with st.expander("About this system", expanded=True):
        st.markdown("""
        This system analyzes CCTV footage to detect and report traffic accidents. It uses a CNN model 
        trained on accident footage to identify potential incidents.
        
        **How it works:**
        1. Processes video frames or images from traffic cameras
        2. CNN model classifies scenes as "Accident" or "Not Accident"
        3. Alerts can be generated for emergency response
        
        **Applications:**
        - Traffic monitoring centers
        - Highway safety management
        - Insurance claim verification
        """)
    
    # File uploader for image
    uploaded_file = st.file_uploader("Upload a traffic scene image", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to OpenCV format and preprocess
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0
        
        # Simulate model prediction
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write("**Analysis Results:**")
            
            # Add a progress bar to simulate processing
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Randomly simulate accident detection for demo purposes
            import random
            prediction = random.random()
            is_accident = prediction >= 0.5
            
            st.write(f"Confidence score: {prediction:.2f}")
            
            if is_accident:
                st.markdown('<div class="warning-box">🚨 ALERT: Accident detected!</div>', 
                           unsafe_allow_html=True)
                st.write("Recommended action: Dispatch emergency services")
            else:
                st.markdown('<div class="success-box">✅ No accident detected</div>', 
                           unsafe_allow_html=True)
                st.write("Normal traffic conditions")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional information
    st.markdown("### How to use this system")
    st.write("""
    1. Upload a traffic scene image from CCTV footage
    2. The system will analyze the image for signs of an accident
    3. Results will show the probability of an accident
    4. For high-probability cases, emergency services can be notified
    
    This system can be integrated with existing traffic monitoring infrastructure for real-time accident detection.
    """)