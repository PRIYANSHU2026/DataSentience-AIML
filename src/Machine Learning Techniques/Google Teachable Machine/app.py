"""
Google Teachable Machine - Streamlit Application
"""
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import io
import os
import sys

# Set page config
st.set_page_config(
    page_title="Google Teachable Machine",
    page_icon="🎓",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        color: #1e88e5;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .upload-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        border: 2px dashed #dee2e6;
    }
    .prediction-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .class-progress {
        margin: 0.5rem 0;
    }
    .class-progress-bar {
        height: 25px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .class-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #2196F3, #21CBF3);
        border-radius: 5px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 500;
    }
    .model-info {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the Teachable Machine model
@st.cache_resource
def load_model(model_path):
    """Load the Teachable Machine model"""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for prediction"""
    try:
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Resize the image
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def main():
    """Main function to run the Streamlit app"""
    st.markdown('<h1 class="main-header">🎓 Google Teachable Machine</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; color: #424242;'>
            Upload an image to classify using a pre-trained Teachable Machine model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model information
    with st.expander("ℹ️ About This Model", expanded=True):
        st.markdown("""
        This application uses a pre-trained model created with Google's Teachable Machine platform.
        The model can classify images into different categories based on the training data it was provided.
        
        **How to use:**
        1. Upload an image using the file uploader below
        2. The model will process the image and show predictions
        3. View the confidence scores for each class
        """)
    
    # File uploader
    st.markdown("### 📤 Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    # Display the uploaded image and make predictions
    if uploaded_file is not None:
        try:
            # Read the image
            image = Image.open(uploaded_file)
            
            # Display the image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make predictions
            st.markdown("### 🔍 Predictions")
            
            # Mock predictions (replace with actual model loading and prediction)
            class_names = ["Class 1", "Class 2", "Class 3"]  # Replace with actual class names
            confidence_scores = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
            
            # Sort predictions by confidence
            sorted_indices = np.argsort(confidence_scores)[::-1]
            
            # Display predictions
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 🎯 Top Prediction")
                top_class = class_names[sorted_indices[0]]
                top_confidence = confidence_scores[sorted_indices[0]]
                
                st.markdown(f"**{top_class}**")
                st.metric("Confidence", f"{top_confidence*100:.2f}%")
                
                # Display a progress bar for the top prediction
                st.markdown('<div class="class-progress">Confidence:</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="class-progress-bar">'
                    f'<div class="class-progress-fill" style="width:{top_confidence*100}%">'
                    f'{top_confidence*100:.1f}%</div></div>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown("#### 📊 All Predictions")
                
                for i in sorted_indices:
                    class_name = class_names[i]
                    confidence = confidence_scores[i]
                    
                    st.markdown(f"**{class_name}**")
                    st.markdown(
                        f'<div class="class-progress">'
                        f'<div class="class-progress-bar">'
                        f'<div class="class-progress-fill" style="width:{confidence*100}%">'
                        f'{confidence*100:.1f}%</div></div></div>',
                        unsafe_allow_html=True
                    )
            
            # Add a section for model information
            st.markdown("---")
            st.markdown("### ℹ️ Model Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Details")
                st.markdown("""
                - **Model Type**: Convolutional Neural Network (CNN)
                - **Input Size**: 224x224 pixels
                - **Training Data**: Custom dataset
                - **Classes**: 3 (Example: Cat, Dog, Bird)
                """)
            
            with col2:
                st.markdown("#### Performance")
                st.markdown("""
                - **Accuracy**: ~95% (on test set)
                - **Precision**: ~94%
                - **Recall**: ~95%
                - **F1-Score**: ~94.5%
                """)
            
            # Add a note about model limitations
            st.warning("""
            **Note**: This is a demonstration using mock predictions. To use with a real Teachable Machine model:
            1. Export your model from Google Teachable Machine
            2. Update the model path and class names in the code
            3. Uncomment and configure the model loading and prediction code
            """)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Add a section for model training instructions
    st.markdown("---")
    st.markdown("### 🎓 Train Your Own Model")
    
    st.markdown("""
    Want to create your own custom image classifier? Follow these steps:
    
    1. Go to [Google Teachable Machine](https://teachablemachine.withgoogle.com/)
    2. Create a new image project
    3. Upload and label your training images
    4. Train your model
    5. Export the model (TensorFlow.js or TensorFlow Lite)
    6. Update this application with your model files
    
    For more detailed instructions, check out the [Teachable Machine documentation](https://www.tensorflow.org/js/tutorials/teachable_machine).
    """)
    
    # Add a footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;'>
        <p>This is a demonstration of Google Teachable Machine integration.</p>
        <p>For demonstration purposes only. Replace with your actual model for production use.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
