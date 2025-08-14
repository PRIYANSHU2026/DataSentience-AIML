"""
Soil Classifier CNN

AI-powered soil type classification using Convolutional Neural Networks.
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

# Add parent directory to path for shared components
sys.path.append(str(Path(__file__).parent.parent.parent))
from ui_components import UIComponents

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model/soil_classifier_cnn.h5")
CLASS_NAMES = [
    'Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil'
]
SOIL_DESCRIPTIONS = {
    'Black Soil': 'Black soil is rich in clay and has high water retention capacity. It is ideal for growing cotton, wheat, and sugarcane.',
    'Cinder Soil': 'Cinder soil is volcanic in origin, porous, and well-draining. It is often used in horticulture and for growing cacti and succulents.',
    'Laterite Soil': 'Laterite soil is rich in iron and aluminum, typically found in tropical regions. It is suitable for tea, coffee, and rubber plantations.',
    'Peat Soil': 'Peat soil is high in organic matter and retains large amounts of water. It is excellent for growing crops that require high moisture.',
    'Yellow Soil': 'Yellow soil is typically found in humid regions and has good drainage. It is suitable for growing rice, wheat, and vegetables.'
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
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = np.expand_dims(image_array, axis=0)
    elif len(image_array.shape) == 2:
        # Convert grayscale to RGB if needed
        image_array = np.stack((image_array,) * 3, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_soil_type(model, image):
    """Make prediction on the input image"""
    try:
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def display_results(image, predicted_class, confidence, all_predictions):
    """Display prediction results with visualizations"""
    st.success("## 🌱 Soil Classification Results")
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Soil Sample", use_column_width=True)
    
    # Display prediction
    soil_type = CLASS_NAMES[predicted_class]
    confidence_percent = confidence * 100
    
    st.markdown(f"### 🌿 Predicted Soil Type: {soil_type}")
    st.write(f"**Confidence:** {confidence_percent:.1f}%")
    
    # Show soil information
    with st.expander("ℹ️ Soil Information"):
        st.write(SOIL_DESCRIPTIONS.get(soil_type, "No additional information available."))
    
    # Show top predictions
    st.markdown("### 📊 Prediction Probabilities")
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Soil Type': CLASS_NAMES,
        'Confidence': all_predictions * 100
    })
    
    # Sort by confidence
    df = df.sort_values('Confidence', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        df,
        x='Confidence',
        y='Soil Type',
        orientation='h',
        title="Soil Type Prediction Probabilities",
        color='Confidence',
        color_continuous_scale='YlOrBr',
        range_x=[0, 100]
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Confidence (%)",
        yaxis_title="Soil Type",
        coloraxis_showscale=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show recommended crops
    with st.expander("🌾 Recommended Crops for This Soil Type"):
        st.write(get_recommended_crops(soil_type))

def get_recommended_crops(soil_type):
    """Get recommended crops for the identified soil type"""
    recommendations = {
        'Black Soil': ["Cotton", "Wheat", "Soybean", "Sunflower", "Millets"],
        'Cinder Soil': ["Cacti", "Succulents", "Lavender", "Rosemary", "Thyme"],
        'Laterite Soil': ["Tea", "Coffee", "Rubber", "Cashew", "Coconut"],
        'Peat Soil': ["Rice", "Cabbage", "Celery", "Onions", "Carrots"],
        'Yellow Soil': ["Rice", "Wheat", "Vegetables", "Fruits", "Pulses"]
    }
    
    crops = recommendations.get(soil_type, ["Various crops can be grown on this soil type."])
    return "• " + "\n• ".join(crops)

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Soil Classifier - Agricultural Solutions",
        page_icon="🌍",
        layout="wide"
    )
    
    # Initialize session state
    if 'model' not in st.session_state:
        with st.spinner("Loading soil classification model..."):
            st.session_state.model = load_model()
    
    # Header
    UIComponents.header(
        "🌍 Soil Classifier CNN",
        "Upload an image of soil to identify its type and get cultivation recommendations"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("🌱 Soil Types")
        st.write("This tool helps you identify soil types from images using AI.")
        
        st.markdown("### 📸 How to Use")
        st.write("1. Upload a clear image of soil")
        st.write("2. The AI will analyze the image")
        st.write("3. View the classification results and recommendations")
        
        if st.button("⬅️ Back to Agricultural Solutions"):
            st.switch_page("../app.py")
        
        st.markdown("---")
        st.markdown("### 🌿 Supported Soil Types")
        st.write("\n".join([f"- {soil}" for soil in CLASS_NAMES]))
    
    # Main content
    st.markdown("### 📤 Upload a Soil Sample Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of soil for classification"
        )
        
        if uploaded_file is not None:
            try:
                # Open and display the image
                image = Image.open(uploaded_file).convert('RGB')
                
                # Make prediction
                if st.button("🔍 Identify Soil Type"):
                    with st.spinner("Analyzing the soil sample..."):
                        # Preprocess the image
                        processed_image = preprocess_image(image)
                        
                        # Make prediction
                        if st.session_state.model is not None:
                            predicted_class, confidence, all_predictions = predict_soil_type(
                                st.session_state.model, 
                                processed_image
                            )
                            
                            if predicted_class is not None:
                                # Display results
                                display_results(image, predicted_class, confidence, all_predictions)
                                
                                # Save results to session state for export
                                st.session_state.last_prediction = {
                                    'image': image,
                                    'soil_type': CLASS_NAMES[predicted_class],
                                    'confidence': confidence,
                                    'all_predictions': all_predictions,
                                    'timestamp': pd.Timestamp.now()
                                }
                        else:
                            st.error("Failed to load the soil classification model.")
                
                # Display the uploaded image
                st.image(image, caption="Uploaded Soil Sample", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.markdown("### ℹ️ Tips for Best Results")
        st.write("""
        - Take a clear, well-lit photo of the soil sample
        - Ensure the soil fills most of the image frame
        - Avoid shadows or glare on the soil surface
        - Use natural lighting when possible
        - Take the photo from directly above the soil
        - Remove any debris or vegetation from the sample
        """)
        
        # Example soil images
        st.markdown("### 🌍 Example Soil Types")
        st.image("https://www.gardeningknowhow.com/wp-content/uploads/2013/06/soil-types.jpg", 
                caption="Common Soil Types", use_column_width=True)
    
    # Export results if available
    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        # Create a DataFrame with the results
        df = pd.DataFrame({
            'Soil Type': [st.session_state.last_prediction['soil_type']],
            'Confidence': [st.session_state.last_prediction['confidence'] * 100],
            'Timestamp': [st.session_state.last_prediction['timestamp']]
        })
        
        # Add top predictions
        top_indices = np.argsort(st.session_state.last_prediction['all_predictions'])[::-1]
        for i, idx in enumerate(top_indices[:3], 1):
            df[f'Top_{i}_Prediction'] = [f"{CLASS_NAMES[idx]} ({st.session_state.last_prediction['all_predictions'][idx]*100:.1f}%)"]
        
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        
        # Download button
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv,
            file_name=f"soil_classification_{st.session_state.last_prediction['timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()