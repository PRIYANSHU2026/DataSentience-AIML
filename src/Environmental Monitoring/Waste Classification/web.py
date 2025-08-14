import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Social links (top-right)
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .social-icons {
            position: absolute;
            top: 15px;
            right: 25px;
        }
        .social-icons a {
            margin: 0 8px;
            text-decoration: none;
            font-size: 1.3rem;
        }
        .social-icons a:hover {
            opacity: 0.8;
        }
    </style>
    <div class="social-icons">
        <a href="https://www.linkedin.com/in/aditya-kumar-3721012aa" target="_blank" style="color:#0e76a8;">
            <i class="fab fa-linkedin"></i>
        </a>
        <a href="https://x.com/kaditya264?s=09" target="_blank" style="color:#1DA1F2;">
            <i class="fab fa-twitter"></i>
        </a>
        <a href="https://github.com/GxAditya" target="_blank" style="color:#333;">
            <i class="fab fa-github"></i>
        </a>
    </div>
""", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("waste_classifier_model.keras")

# Prediction function
def predict_fun(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (96, 96))
    img = img / 255.0
    img = np.reshape(img, [-1, 96, 96, 3])
    result = model.predict(img)[0][0]
    if result < 0.5:
        return 'The Image Shown is Organic Waste'
    else:
        return 'The Image Shown is Recyclable Waste'

# Page config
st.set_page_config(page_title="Waste Classification App", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@400;700&display=swap');

        html, body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppContainer"],
        [data-testid="stAppMain"] {
            background: linear-gradient(135deg, #a8edea, #fed6e3, #c3fdb8, #89d4cf);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            font-family: 'Caveat', cursive;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .main > div {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h3 { text-align: center; color: #1b5e20; }
        .info-text {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #2e2e2e;
            text-align: center;
            background: rgba(255,255,255,0.9);
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .stButton > button {
            background: linear-gradient(90deg, #43a047, #2e7d32);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #2e7d32, #43a047);
            transform: scale(1.05);
        }
        div[data-testid="stFileUploader"] {
            border: 2px dashed #1e88e5;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            background: white;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
        .prediction {
            font-size: 1.3rem;
            font-weight: bold;
            text-align: center;
            padding: 12px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .organic { color: #2e7d32; background: #e8f5e9; }
        .recyclable { color: #1565c0; background: #e3f2fd; }
        .app-icon {
            display: block;
            margin: 0 auto 15px;
            width: 120px;
            height: auto;
            border-radius: 50%;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Page state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Home page
if st.session_state.page == "Home":
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://raw.githubusercontent.com/GxAditya/Waste-Classification/main/waste.png' class='app-icon'>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h1>Waste Classification Using Deep Learning</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>This project uses a CNN to classify waste as either Organic or Recyclable. It helps in better waste management for a cleaner environment.</p>", unsafe_allow_html=True)

    st.markdown("<h3>How it Works</h3>", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-text'>
        1️⃣ Upload an image of waste material.<br>
        2️⃣ The model analyzes it and classifies it.<br>
        3️⃣ Helps in proper disposal and recycling.
        </div>
    """, unsafe_allow_html=True)

    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("Try It Now!", use_container_width=True):
            st.session_state.page = "Classification"
            st.rerun()

# Classification page
elif st.session_state.page == "Classification":
    st.markdown("<h1>Waste Classification</h1>", unsafe_allow_html=True)

    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()

    st.markdown("<p style='text-align: center;'>Upload an image to classify it as Organic or Recyclable.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Classify Image", use_container_width=True):
                result = predict_fun(image)
                if "Organic" in result:
                    st.markdown(f"<div class='prediction organic'>{result}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='prediction recyclable'>{result}</div>", unsafe_allow_html=True)
