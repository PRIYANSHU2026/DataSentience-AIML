"""
Plant Seedlings Classification

AI-powered classification of plant seedlings using deep learning.
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
import cv2

# Add parent directory to path for shared components
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui_components import UIComponents

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model/plant_seedlings_model.pth")
CLASS_NAMES = [
    'Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
    'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
    'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet'
]

# Define the model architecture (ResNet18 based on the notebook)
class PlantSeedlingsModel(nn.Module):
    def __init__(self, num_classes=12):
        super(PlantSeedlingsModel, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

def load_model():
    """Load the trained model"""
    try:
        model = PlantSeedlingsModel(num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the image for model inference"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_seedling(model, image):
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
    st.success("## 🌱 Seedling Classification Results")
    
    # Display the uploaded image with bounding box
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Display prediction
    plant_name = CLASS_NAMES[predicted_class]
    confidence = probabilities[predicted_class] * 100
    
    st.markdown(f"### 🌿 Predicted Plant: {plant_name}")
    st.write(f"**Confidence:** {confidence:.1f}%")
    
    # Show plant information
    with st.expander("ℹ️ Plant Information"):
        st.write(get_plant_information(plant_name))
    
    # Show top predictions
    st.markdown("### 📊 Prediction Probabilities")
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_classes = [CLASS_NAMES[i] for i in top_indices]
    top_probs = [probabilities[i] * 100 for i in top_indices]
    
    fig = px.bar(
        x=top_probs,
        y=top_classes,
        orientation='h',
        labels={'x': 'Confidence (%)', 'y': 'Plant Species'},
        title=f"Top {top_k} Predictions",
        color=top_probs,
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig, use_container_width=True)

def get_plant_information(plant_name):
    """Get information about the identified plant"""
    info = {
        'Black-grass': 'Common weed in cereals, can significantly reduce crop yields.',
        'Charlock': 'Wild mustard plant, considered a weed in many crops.',
        'Cleavers': 'Also known as "sticky weed", common in hedgerows and crops.',
        'Common Chickweed': 'Widespread weed that can form dense mats in crops.',
        'Common wheat': 'One of the most important cereal crops worldwide.',
        'Fat Hen': 'Fast-growing weed that competes with crops for nutrients.',
        'Loose Silky-bent': 'Common weed in cereal crops, particularly wheat.',
        'Maize': 'Major cereal crop, also known as corn in North America.',
        'Scentless Mayweed': 'Common weed in arable land, similar to chamomile.',
        'Shepherds Purse': 'Widespread weed with distinctive heart-shaped seed pods.',
        'Small-flowered Cranesbill': 'Common weed in arable fields and gardens.',
        'Sugar beet': 'Important crop for sugar production.'
    }
    return info.get(plant_name, "No additional information available.")

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Plant Seedlings Classification - Agricultural Solutions",
        page_icon="🌿",
        layout="wide"
    )
    
    # Initialize session state
    if 'model' not in st.session_state:
        with st.spinner("Loading plant seedlings classification model..."):
            st.session_state.model = load_model()
    
    # Header
    UIComponents.header(
        "🔍 Plant Seedlings Classification",
        "Upload an image of a plant seedling to identify its species"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("🌿 Plant Seedlings")
        st.write("This tool helps you identify plant species from seedling images using AI.")
        
        st.markdown("### 📸 How to Use")
        st.write("1. Upload a clear image of a plant seedling")
        st.write("2. The AI will analyze the image")
        st.write("3. View the classification results")
        
        if st.button("⬅️ Back to Agricultural Solutions"):
            st.switch_page("../app.py")
        
        st.markdown("---")
        st.markdown("### 🌱 Supported Plants")
        st.write("\n".join([f"- {plant}" for plant in CLASS_NAMES]))
    
    # Main content
    st.markdown("### 📤 Upload a Seedling Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant seedling for classification"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            try:
                image = Image.open(uploaded_file).convert('RGB')
                
                # Make prediction
                if st.button("🔍 Identify Plant"):
                    with st.spinner("Analyzing the image..."):
                        # Preprocess the image
                        input_tensor = preprocess_image(image)
                        
                        # Make prediction
                        if st.session_state.model is not None:
                            predicted_class, probabilities = predict_seedling(
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
                                    'probabilities': probabilities,
                                    'plant_name': CLASS_NAMES[predicted_class]
                                }
                        else:
                            st.error("Failed to load the classification model.")
                
                # Display the uploaded image
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.markdown("### ℹ️ Tips for Best Results")
        st.write("""
        - Use a clear, well-lit image of a single seedling
        - Place the seedling on a plain, contrasting background
        - Ensure the entire seedling is visible
        - Avoid shadows or glare on the leaves
        - Take the photo from directly above the seedling
        """)
        
        # Example images
        st.markdown("### 🌿 Example Images")
        st.image("https://storage.googleapis.com/kaggle-media/competitions/plant-seedlings/plant-seedlings.jpg", 
                caption="Example Seedlings", use_column_width=True)
    
    # Export results if available
    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        # Create a DataFrame with the results
        df = pd.DataFrame({
            'Plant Species': [st.session_state.last_prediction['plant_name']],
            'Confidence': [st.session_state.last_prediction['probabilities'][st.session_state.last_prediction['predicted_class']] * 100],
            'Timestamp': [pd.Timestamp.now()]
        })
        
        # Add top predictions
        top_indices = np.argsort(st.session_state.last_prediction['probabilities'])[::-1][:5]
        for i, idx in enumerate(top_indices, 1):
            df[f'Top_{i}_Prediction'] = [f"{CLASS_NAMES[idx]} ({st.session_state.last_prediction['probabilities'][idx]*100:.1f}%)"]
        
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        
        # Download button
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv,
            file_name=f"seedling_classification_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()