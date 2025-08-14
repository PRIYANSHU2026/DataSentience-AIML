"""
Waste Classification - Streamlit App
"""
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Waste Classification",
    page_icon="🗑️",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the pre-trained waste classification model"""
    try:
        model_path = Path(__file__).parent / "waste_classifier_model.keras"
        if not model_path.exists():
            st.error(f"Model file not found at: {model_path}")
            return None
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_waste(image, model):
    """Classify waste image as Organic or Recyclable"""
    try:
        # Convert PIL Image to OpenCV format and preprocess
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (96, 96))  # Match model's expected sizing
        img = img / 255.0  # Normalize pixel values
        img = np.reshape(img, [1, 96, 96, 3])
        
        # Make prediction
        prediction = model.predict(img)[0][0]
        
        # Determine class and confidence
        if prediction < 0.5:
            return "Organic Waste", 1.0 - prediction
        else:
            return "Recyclable Waste", prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def main():
    st.title("🗑️ Waste Classification")
    st.markdown("""
    Classify waste as Organic or Recyclable using deep learning.
    Upload an image of waste material to get started.
    """)
    
    # Load model
    with st.spinner("Loading waste classification model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load the waste classification model. Please check the model file.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image of waste material"
    )
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            if st.button("Classify Waste", type="primary"):
                with st.spinner("Analyzing waste..."):
                    result, confidence = predict_waste(image, model)
                    
                    if result is not None:
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Prediction")
                            if result == "Organic Waste":
                                st.success(f"🍂 **{result}**")
                            else:
                                st.info(f"♻️ **{result}**")
                            
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        with col2:
                            # Create a simple gauge chart for confidence
                            fig, ax = plt.subplots(figsize=(6, 1))
                            ax.barh([0], [confidence*100], height=0.5, 
                                   color='#4CAF50' if result == "Organic Waste" else '#2196F3')
                            ax.set_xlim(0, 100)
                            ax.set_xticks([0, 50, 100])
                            ax.set_xticklabels(['0%', '50%', '100%'])
                            ax.set_yticks([])
                            ax.set_title("Confidence Level")
                            st.pyplot(fig, use_container_width=True)
                        
                        # Show disposal recommendations
                        st.markdown("### Disposal Recommendation")
                        if result == "Organic Waste":
                            st.info("""
                            **For Organic Waste:**
                            - Compost if possible
                            - Use a green bin if available
                            - Keep separate from recyclables
                            - Consider home composting for food scraps
                            """)
                        else:
                            st.info("""
                            **For Recyclable Waste:**
                            - Rinse containers before recycling
                            - Check local recycling guidelines
                            - Remove any non-recyclable components
                            - Flatten boxes to save space
                            """)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        # Show sample images
        st.markdown("### Example Images")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(
                "https://images.unsplash.com/photo-1606787366850-de6330128bfc?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80",
                caption="Organic Waste Example",
                use_column_width=True
            )
        
        with col2:
            st.image(
                "https://images.unsplash.com/photo-1587844105205-329fe9b0b5d7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80",
                caption="Recyclable Waste Example",
                use_column_width=True
            )
        
        st.info("ℹ️ Upload an image of waste material to classify it as Organic or Recyclable.")

if __name__ == "__main__":
    main()
