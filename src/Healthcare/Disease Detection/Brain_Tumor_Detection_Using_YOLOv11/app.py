"""
Brain Tumor Detection - Streamlit App
"""
import os
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

class BrainTumorDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or self._get_default_model_path()
        self.load_model()
    
    def _get_default_model_path(self):
        """Get the default path to the YOLO model"""
        # This should point to your trained model weights
        return "yolo11/runs/detect/train2/weights/best.pt"
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            from ultralytics import YOLO
            if not os.path.exists(self.model_path):
                st.error(f"Model file not found at: {self.model_path}")
                return False
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, image_path):
        """Run prediction on an image"""
        if self.model is None:
            st.error("Model not loaded. Please check the model path.")
            return None
        
        try:
            # Run inference
            results = self.model(image_path)
            
            # Save results
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the result image
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            results[0].save(filename=output_path)
            
            return {
                'output_path': output_path,
                'predictions': results[0].boxes.data.cpu().numpy(),
                'names': results[0].names
            }
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None

def main():
    st.title("🧠 Brain Tumor Detection")
    st.markdown("""
    Detect brain tumors in MRI scans using YOLOv11 deep learning model.
    Upload an MRI scan to detect potential tumors.
    """)
    
    # Initialize model
    detector = BrainTumorDetector()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an MRI scan...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI scan in JPG, JPEG, or PNG format"
    )
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            
            # Save the uploaded image temporarily
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            image.save(temp_path)
            
            # Run detection when button is clicked
            if st.button("Detect Tumor", type="primary"):
                with st.spinner("Analyzing MRI scan..."):
                    result = detector.predict(temp_path)
                    
                    if result and os.path.exists(result['output_path']):
                        # Display the detection result
                        st.markdown("---")
                        st.subheader("Detection Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Show the original image
                            st.image(
                                temp_path,
                                caption="Original Scan",
                                use_column_width=True
                            )
                        
                        with col2:
                            # Show the detection result
                            st.image(
                                result['output_path'],
                                caption="Tumor Detection",
                                use_column_width=True
                            )
                        
                        # Show detection summary
                        num_detections = len(result['predictions'])
                        if num_detections > 0:
                            st.warning(f"⚠️ **{num_detections} potential tumor(s) detected!**")
                            
                            # Show confidence scores for each detection
                            for i, detection in enumerate(result['predictions']):
                                x1, y1, x2, y2, confidence, class_id = detection
                                class_name = result['names'].get(int(class_id), 'tumor')
                                st.write(f"- Detected {class_name} with {confidence*100:.1f}% confidence")
                            
                            st.info("""
                            **Recommendations:**
                            - Consult with a medical professional for further evaluation
                            - Share these results with your healthcare provider
                            - Schedule a follow-up appointment if recommended
                            """)
                        else:
                            st.success("✅ No tumors detected in the scan.")
                            st.info("""
                            **Note:** This is a screening tool, not a diagnosis. 
                            Always consult with a healthcare professional for medical advice.
                            """)
                    else:
                        st.error("Failed to process the MRI scan. Please try again.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please make sure you've uploaded a valid brain MRI scan.")
    else:
        # Show example MRI scans
        st.markdown("### Example MRI Scans")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(
                "https://images.unsplash.com/photo-1631549916768-4119b2e5f926?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80",
                caption="Normal Brain MRI",
                use_column_width=True
            )
        
        with col2:
            st.image(
                "https://images.unsplash.com/photo-1631549916647-3e2e6d4a1f2b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80",
                caption="MRI Showing Brain Tumor",
                use_column_width=True
            )
        
        st.info("ℹ️ Upload a brain MRI scan to detect potential tumors.")

if __name__ == "__main__":
    main()
