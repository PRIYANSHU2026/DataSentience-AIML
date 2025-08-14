"""
Fire Detection - Streamlit App
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Fire Detection",
    page_icon="🔥",
    layout="centered"
)

def detect_fire(image):
    """Detect fire in the given image using computer vision techniques"""
    try:
        # Convert PIL Image to OpenCV format
        img = np.array(image)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # Threshold the image
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the original image
        output = img.copy()
        fire_detected = False
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small contours
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(output, 'Fire Detected', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                fire_detected = True
        
        return output, fire_detected
    
    except Exception as e:
        st.error(f"Error in fire detection: {str(e)}")
        return None, False

def main():
    st.title("🔥 Fire Detection")
    st.markdown("Upload an image to detect fire using computer vision techniques.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Detect Fire"):
            with st.spinner("Analyzing image for fire..."):
                # Process the image
                result, fire_detected = detect_fire(image)
                
                if result is not None:
                    # Display result
                    st.image(result, caption="Fire Detection Result", use_column_width=True)
                    
                    if fire_detected:
                        st.error("🔥 Fire detected in the image!")
                    else:
                        st.success("✅ No fire detected in the image.")
                else:
                    st.warning("Could not process the image. Please try another one.")

if __name__ == "__main__":
    main()
