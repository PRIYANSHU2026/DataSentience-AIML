"""
Plant Disease Detection System

AI-powered plant disease detection using deep learning models.
"""
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Add parent directory to path for shared components
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui_components import UIComponents

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant-disease-model-complete.pth")
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Define the model architecture (should match the trained model)
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseModel, self).__init__()
        # This is a simplified version - should match your actual model architecture
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Load the trained model
def load_model():
    """Load the trained model"""
    try:
        model = PlantDiseaseModel(num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing
def preprocess_image(image):
    """Preprocess the image for model inference"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Make prediction
def predict_disease(model, image):
    """Make prediction on the input image"""
    try:
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return predicted.item(), probabilities[0].numpy()
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def display_results(image, predicted_class, probabilities, top_k=5):
    """Display prediction results with visualizations"""
    st.success("## 🌿 Disease Detection Results")
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Display prediction
    disease_name = CLASS_NAMES[predicted_class].replace('_', ' ').title()
    confidence = probabilities[predicted_class] * 100
    
    if 'healthy' in disease_name.lower():
        st.balloons()
        st.success(f"### 🍃 Plant is Healthy!")
        st.write(f"**Confidence:** {confidence:.1f}%")
    else:
        st.warning(f"### ⚠️ Detected: {disease_name}")
        st.write(f"**Confidence:** {confidence:.1f}%")
        
        # Show treatment suggestions
        with st.expander("💊 Treatment Recommendations"):
            st.write(get_treatment_suggestions(disease_name))
    
    # Show top predictions
    st.markdown("### 📊 Prediction Probabilities")
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_classes = [CLASS_NAMES[i].replace('_', ' ').title() for i in top_indices]
    top_probs = [probabilities[i] * 100 for i in top_indices]
    
    fig = px.bar(
        x=top_probs,
        y=top_classes,
        orientation='h',
        labels={'x': 'Confidence (%)', 'y': 'Disease'},
        title=f"Top {top_k} Predictions",
        color=top_probs,
        color_continuous_scale='Teal'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show prevention tips
    with st.expander("🌱 Prevention Tips"):
        st.write(get_prevention_tips(disease_name))

def get_treatment_suggestions(disease_name):
    """Get treatment suggestions based on the detected disease"""
    treatments = {
        'apple scab': "Apply fungicides in early spring. Remove and destroy infected leaves.",
        'black rot': 'Prune infected branches. Apply appropriate fungicides during the growing season.',
        'powdery mildew': 'Improve air circulation. Apply sulfur or potassium bicarbonate-based fungicides.',
        'leaf spot': 'Remove infected leaves. Apply copper-based fungicides.',
        'blight': 'Remove and destroy infected plants. Apply appropriate fungicides.',
        'rust': 'Remove infected leaves. Apply fungicides containing myclobutanil or tebuconazole.',
        'mosaic virus': 'Remove and destroy infected plants. Control aphid populations.'
    }
    
    disease_lower = disease_name.lower()
    for key in treatments:
        if key in disease_lower:
            return treatments[key]
    
    return "Consult with a local agricultural extension service for specific treatment recommendations."

def get_prevention_tips(disease_name):
    """Get prevention tips based on the detected disease"""
    tips = [
        "🌱 Plant disease-resistant varieties when available.",
        "💧 Water plants at the base to keep foliage dry.",
        "🌞 Ensure proper spacing between plants for good air circulation.",
        "🧤 Clean tools between uses to prevent disease spread.",
        "🍂 Remove and destroy infected plant debris at the end of the season.",
        "🌦️ Monitor plants regularly for early signs of disease.",
        "🌿 Use proper fertilization to maintain plant health.",
        "🔄 Rotate crops to prevent disease buildup in the soil."
    ]
    
    if 'blight' in disease_name.lower():
        tips.append("🌧️ Avoid overhead watering to prevent the spread of fungal spores.")
    
    if 'mildew' in disease_name.lower():
        tips.append("💨 Ensure good air circulation around plants to reduce humidity.")
    
    if 'virus' in disease_name.lower():
        tips.append("🐜 Control insect vectors that may spread viral diseases.")
    
    return "\n\n".join([f"- {tip}" for tip in tips])

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Plant Disease Detection - Agricultural Solutions",
        page_icon="🌱",
        layout="wide"
    )
    
    # Initialize session state
    if 'model' not in st.session_state:
        with st.spinner("Loading plant disease detection model..."):
            st.session_state.model = load_model()
    
    # Header
    UIComponents.header(
        "🔍 Plant Disease Detection",
        "Upload an image of a plant leaf to detect potential diseases"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("🌿 Plant Disease Detection")
        st.write("This tool helps you identify plant diseases from leaf images using AI.")
        
        st.markdown("### 📸 How to Use")
        st.write("1. Upload a clear image of a plant leaf")
        st.write("2. The AI will analyze the image")
        st.write("3. View the results and recommendations")
        
        if st.button("⬅️ Back to Agricultural Solutions"):
            st.switch_page("../app.py")
        
        st.markdown("---")
        st.markdown("### 🌱 Supported Plants")
        st.write("""
        - Apple
        - Blueberry
        - Cherry
        - Corn
        - Grape
        - Orange
        - Peach
        - Pepper
        - Potato
        - Raspberry
        - Soybean
        - Squash
        - Strawberry
        - Tomato
        """)
    
    # Main content
    st.markdown("### 📤 Upload a Plant Leaf Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant leaf for disease detection"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            try:
                image = Image.open(uploaded_file).convert('RGB')
                
                # Make prediction
                if st.button("🔍 Detect Disease"):
                    with st.spinner("Analyzing the image..."):
                        # Preprocess the image
                        input_tensor = preprocess_image(image)
                        
                        # Make prediction
                        if st.session_state.model is not None:
                            predicted_class, probabilities = predict_disease(
                                st.session_state.model, 
                                input_tensor
                            )
                            
                            if predicted_class is not None:
                                # Display results
                                display_results(image, predicted_class, probabilities)
                                
                                # Save results to session state for export
                                st.session_state.last_prediction = {
                                    'image': image,
                                    'predicted_class': predicted_class,
                                    'probabilities': probabilities
                                }
                        else:
                            st.error("Failed to load the disease detection model.")
                
                # Display the uploaded image
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.markdown("### ℹ️ Tips for Best Results")
        st.write("""
        - Use a clear, well-lit image of a single leaf
        - Place the leaf on a plain, contrasting background
        - Ensure the entire leaf is visible
        - Avoid shadows or glare on the leaf
        - Take the photo from directly above the leaf
        """)
        
        # Example images
        st.markdown("### 🌿 Example Images")
        st.image("https://raw.githubusercontent.com/ashishpatel26/Plant-Disease-Detection/master/asset/healthy.jpg", 
                caption="Healthy Leaf", use_column_width=True)
        st.image("https://raw.githubusercontent.com/ashishpatel26/Plant-Disease-Detection/master/asset/diseased.jpg", 
                caption="Diseased Leaf", use_column_width=True)
    
    # Export results if available
    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        # Create a DataFrame with the results
        df = pd.DataFrame({
            'Disease': [CLASS_NAMES[st.session_state.last_prediction['predicted_class']].replace('_', ' ').title()],
            'Confidence': [st.session_state.last_prediction['probabilities'][st.session_state.last_prediction['predicted_class']] * 100],
            'Timestamp': [pd.Timestamp.now()]
        })
        
        # Add top predictions
        top_indices = np.argsort(st.session_state.last_prediction['probabilities'])[::-1][:5]
        for i, idx in enumerate(top_indices, 1):
            df[f'Top_{i}_Prediction'] = [f"{CLASS_NAMES[idx].replace('_', ' ').title()} ({st.session_state.last_prediction['probabilities'][idx]*100:.1f}%)"]
        
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        
        # Download button
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv,
            file_name=f"plant_detection_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()