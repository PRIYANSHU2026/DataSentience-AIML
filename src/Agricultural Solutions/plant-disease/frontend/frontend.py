import streamlit as st
import requests
from PIL import Image
import io


API_URL = "http://127.0.0.1:8000/predict" 

st.set_page_config(page_title="Smart Plant AI", page_icon="🌱")

# UI Styling
st.title(" Smart Plant Disease Detector")
st.markdown("---")
st.caption("Tech Stack:** FastAPI (Backend) + Streamlit (Frontend) + MobileNet")

# File Uploader
uploaded_file = st.file_uploader("Upload a leaf image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image Display
    image = Image.open(uploaded_file)
    st.image(image, caption=' Selected Leaf Image', use_container_width=True)

    # Prediction Button
    if st.button('🔍 Analyze Plant Health'):
        with st.spinner('AI is analyzing the leaf... Please wait.'):
            try:
                # 1. Reset file pointer for multiple reads
                uploaded_file.seek(0)
                
                # 2. Files dictionary (Backend mein 'file' key check karega)
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
               
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    label = result['class']
                    conf = result['confidence']

                    # Result Display
                    st.markdown("### Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"**Disease:** \n\n {label}")
                    with col2:
                        st.info(f"**Confidence:** \n\n {conf:.2f}%")
                    
                    # Basic Advice logic (Optional)
                    if conf < 60:
                        st.warning("Confidence low hai. Kripya clear photo upload karein.")

                else:
                    st.error(f" Server Error: {response.status_code}")
                    st.write(response.text)

            except requests.exceptions.ConnectionError:
                st.error("Backend Server (FastAPI) connect nahi ho pa raha. Check karein ki 'python app.py' chal raha hai.")
            except Exception as e:
                st.error(f"Unexpected Error: {e}")

st.markdown("---")
st.info(" **Tip:** Behtar result ke liye leaf ko light mein aur plain background par scan karein.")