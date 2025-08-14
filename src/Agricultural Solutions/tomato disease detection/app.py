"""
Tomato Disease Detection

AI-powered detection of tomato plant diseases from leaf images.
"""
import os
import sys
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import plotly.express as px
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Add parent directory to path for shared components
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui_components import UIComponents

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model/tomato_disease_model.h5")
CLASS_NAMES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Disease information
DISEASE_INFO = {
    'Tomato___Bacterial_spot': {
        'name': 'Bacterial Spot',
        'description': 'Caused by Xanthomonas bacteria, this disease causes small, dark, water-soaked spots on leaves that may have a yellow halo.',
        'treatment': 'Use copper-based bactericides, practice crop rotation, and remove infected plants.'
    },
    'Tomato___Early_blight': {
        'name': 'Early Blight',
        'description': 'Caused by the fungus Alternaria solani, it creates target-like dark spots with concentric rings on leaves.',
        'treatment': 'Apply fungicides, ensure good air circulation, and remove infected leaves.'
    },
    'Tomato___Late_blight': {
        'name': 'Late Blight',
        'description': 'Caused by Phytophthora infestans, it causes large, dark lesions on leaves and stems with white fungal growth under humid conditions.',
        'treatment': 'Apply appropriate fungicides, remove and destroy infected plants, and avoid overhead watering.'
    },
    'Tomato___Leaf_Mold': {
        'name': 'Leaf Mold',
        'description': 'Caused by the fungus Passalora fulva, it causes pale green or yellowish spots on upper leaf surfaces with olive-green to brown velvety growth underneath.',
        'treatment': 'Improve air circulation, reduce humidity, and apply appropriate fungicides.'
    },
    'Tomato___Septoria_leaf_spot': {
        'name': 'Septoria Leaf Spot',
        'description': 'Caused by the fungus Septoria lycopersici, it creates small, circular spots with dark brown margins and light gray centers on leaves.',
        'treatment': 'Remove infected leaves, apply fungicides, and avoid overhead watering.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'name': 'Spider Mites',
        'description': 'Tiny arachnids that feed on plant sap, causing stippling, yellowing, and webbing on leaves.',
        'treatment': 'Use insecticidal soaps, neem oil, or miticides. Increase humidity to deter mites.'
    },
    'Tomato___Target_Spot': {
        'name': 'Target Spot',
        'description': 'Caused by the fungus Corynespora cassiicola, it creates small, dark spots with light centers that resemble targets.',
        'treatment': 'Apply appropriate fungicides and remove infected plant debris.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'name': 'Tomato Yellow Leaf Curl Virus',
        'description': 'A viral disease transmitted by whiteflies, causing upward curling of leaves, yellowing, and stunted growth.',
        'treatment': 'Control whitefly populations, remove infected plants, and use resistant varieties.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'name': 'Tomato Mosaic Virus',
        'description': 'A viral disease causing mottled light and dark green patterns on leaves, leaf distortion, and stunted growth.',
        'treatment': 'Remove and destroy infected plants, control weeds, and practice good sanitation.'
    },
    'Tomato___healthy': {
        'name': 'Healthy',
        'description': 'The tomato plant shows no signs of disease and appears to be growing normally.',
        'treatment': 'Continue current care practices including proper watering, fertilization, and monitoring for pests.'
    }
}

def load_model():
    """Load the trained CNN model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the image for model inference"""
    # Resize and normalize the image
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = np.expand_dims(image_array, axis=0)
    elif len(image_array.shape) == 2:
        # Convert grayscale to RGB if needed
        image_array = np.stack((image_array,) * 3, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_disease(model, image):
    """Make prediction on the input image"""
    try:
        predictions = model.predict(image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        return CLASS_NAMES[predicted_class_idx], confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def display_results(image, disease_class, confidence, all_predictions):
    """Display prediction results with visualizations"""
    st.success("## 🍅 Tomato Disease Detection Results")
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Tomato Leaf", use_column_width=True)
    
    # Get disease information
    disease_info = DISEASE_INFO.get(disease_class, {
        'name': disease_class.replace('Tomato___', '').replace('_', ' '),
        'description': 'No additional information available.',
        'treatment': 'Consult with an agricultural expert for specific treatment options.'
    })
    
    # Display prediction
    st.markdown(f"### 🌱 **{disease_info['name']}**")
    st.write(f"**Confidence:** {confidence*100:.1f}%")
    
    # Show disease information
    with st.expander("ℹ️ Disease Information"):
        st.write(disease_info['description'])
    
    # Show treatment information
    if disease_class != 'Tomato___healthy':
        with st.expander("💊 Recommended Treatment"):
            st.write(disease_info['treatment'])
    
    # Show top predictions
    st.markdown("### 📊 Prediction Probabilities")
    
    # Create a DataFrame for visualization
    pred_df = pd.DataFrame({
        'Disease': [cls.replace('Tomato___', '').replace('_', ' ') for cls in CLASS_NAMES],
        'Confidence': all_predictions * 100
    })
    
    # Sort by confidence
    pred_df = pred_df.sort_values('Confidence', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        pred_df.head(5),  # Show top 5 predictions
        x='Confidence',
        y='Disease',
        orientation='h',
        title="Top 5 Predictions",
        color='Confidence',
        color_continuous_scale='Reds',
        range_x=[0, 100]
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Confidence (%)",
        yaxis_title="Disease",
        coloraxis_showscale=False,
        height=400
    )
    
    st.plotly_chart(fig, use_column_width=True)

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Tomato Disease Detection - Agricultural Solutions",
        page_icon="🍅",
        layout="wide"
    )
    
    # Initialize session state
    if 'model' not in st.session_state:
        with st.spinner("Loading tomato disease detection model..."):
            st.session_state.model = load_model()
    
    # Header
    UIComponents.header(
        "🍅 Tomato Disease Detection",
        "Upload an image of a tomato leaf to detect diseases and get treatment recommendations"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("🍃 Tomato Diseases")
        st.write("This tool helps you identify common tomato plant diseases from leaf images using AI.")
        
        st.markdown("### 📸 How to Use")
        st.write("1. Take a clear photo of a tomato leaf")
        st.write("2. Upload the image using the file uploader")
        st.write("3. Click 'Detect Disease' to analyze the image")
        st.write("4. View the results and treatment recommendations")
        
        if st.button("⬅️ Back to Agricultural Solutions"):
            st.switch_page("../app.py")
        
        st.markdown("---")
        st.markdown("### 🌿 Supported Diseases")
        for disease in CLASS_NAMES:
            if disease != 'Tomato___healthy':
                st.write(f"- {disease.replace('Tomato___', '').replace('_', ' ')}")
    
    # Main content
    st.markdown("### 📤 Upload a Tomato Leaf Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a tomato leaf for disease detection"
        )
        
        if uploaded_file is not None:
            try:
                # Open and display the image
                image = Image.open(uploaded_file).convert('RGB')
                
                # Make prediction
                if st.button("🔍 Detect Disease"):
                    with st.spinner("Analyzing the tomato leaf..."):
                        # Preprocess the image
                        processed_image = preprocess_image(image)
                        
                        # Make prediction
                        if st.session_state.model is not None:
                            disease_class, confidence, all_predictions = predict_disease(
                                st.session_state.model, 
                                processed_image
                            )
                            
                            if disease_class is not None:
                                # Display results
                                display_results(image, disease_class, confidence, all_predictions)
                                
                                # Save results to session state for export
                                st.session_state.last_prediction = {
                                    'image': image,
                                    'disease': disease_class,
                                    'confidence': confidence,
                                    'all_predictions': all_predictions,
                                    'timestamp': pd.Timestamp.now()
                                }
                        else:
                            st.error("Failed to load the disease detection model.")
                
                # Display the uploaded image
                st.image(image, caption="Uploaded Tomato Leaf", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.markdown("### ℹ️ Tips for Best Results")
        st.write("""
        - Take photos in natural daylight
        - Ensure the leaf fills most of the frame
        - Place the leaf on a plain, contrasting background
        - Avoid shadows or glare on the leaf
        - Take multiple photos from different angles
        - Focus on the most affected areas of the leaf
        - Include both healthy and diseased parts if possible
        """)
        
        # Example images
        st.markdown("### 🌿 Example Leaf Conditions")
        st.image("https://www.gardeningknowhow.com/wp-content/uploads/2021/06/tomato-leaf-problems-400x300.jpg", 
                caption="Common Tomato Leaf Conditions", use_column_width=True)
    
    # Export results if available
    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        # Create a DataFrame with the results
        df = pd.DataFrame({
            'Disease': [st.session_state.last_prediction['disease'].replace('Tomato___', '').replace('_', ' ')],
            'Confidence': [st.session_state.last_prediction['confidence'] * 100],
            'Timestamp': [st.session_state.last_prediction['timestamp']]
        })
        
        # Add top predictions
        top_indices = np.argsort(st.session_state.last_prediction['all_predictions'])[::-1][:3]
        for i, idx in enumerate(top_indices, 1):
            disease_name = CLASS_NAMES[idx].replace('Tomato___', '').replace('_', ' ')
            df[f'Top_{i}_Prediction'] = [f"{disease_name} ({st.session_state.last_prediction['all_predictions'][idx]*100:.1f}%)"]
        
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        
        # Download button
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv,
            file_name=f"tomato_disease_{st.session_state.last_prediction['timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()