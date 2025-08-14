"""
Disease Detection - Streamlit Application
"""
import streamlit as st
import os
import sys
from pathlib import Path
import importlib.util

# Set page config
st.set_page_config(
    page_title="Disease Detection",
    page_icon="🩺",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1565c0;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
    }
    .tool-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #1976d2;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    .tool-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .footer {
        margin-top: 2rem;
        padding: 1rem;
        text-align: center;
        color: #757575;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for tool selection
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = None

def tool_card(title, description, icon, button_text, key, available=True):
    """Create a tool card component"""
    with st.container():
        st.markdown(f"""
        <div class="tool-card" style="opacity: {"0.6" if not available else "1"};">
            <h3>{icon} {title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if available:
            if st.button(button_text, key=key, use_container_width=True):
                st.session_state.selected_tool = key
                st.rerun()
        else:
            st.button(button_text, key=key, use_container_width=True, disabled=True)

def load_tool(tool_name):
    """Load and run the selected tool"""
    st.markdown("---")
    
    # Back button
    if st.button("← Back to Tools", key=f"back_{tool_name}"):
        go_back_to_main()
        return
    
    # Tool content area
    if tool_name == "brain_tumor_detection":
        load_brain_tumor_detection()
    elif tool_name == "cancer_detection":
        load_cancer_detection()
    elif tool_name == "lung_cancer_prediction":
        load_lung_cancer_prediction()
    elif tool_name == "alzheimers_prediction":
        load_alzheimers_prediction()
    elif tool_name == "diabetes_prediction":
        load_diabetes_prediction()
    else:
        st.warning("This tool is not yet available.")
        st.button("← Back to Tools", key=f"back_from_{tool_name}", on_click=lambda: setattr(st.session_state, 'selected_tool', None))

def load_brain_tumor_detection():
    """Load the Brain Tumor Detection tool"""
    try:
        brain_tumor_path = os.path.join(os.path.dirname(__file__), "Brain_Tumor_Detection_Using_YOLOv11")
        if brain_tumor_path not in sys.path:
            sys.path.append(brain_tumor_path)
        
        # Try to import the module
        try:
            from app import main as brain_tumor_main
            st.markdown("## 🧠 Brain Tumor Detection")
            st.markdown("Upload an MRI scan to detect brain tumors.")
            brain_tumor_main()
        except ImportError as e:
            st.error(f"Failed to import Brain Tumor Detection module: {str(e)}")
            st.info(f"Python path: {sys.path}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the Brain Tumor Detection module is properly set up.")

def load_alzheimers_prediction():
    """Load the Alzheimer's Prediction tool"""
    st.info("Alzheimer's Prediction tool is coming soon!")
    # TODO: Implement Alzheimer's prediction integration

def load_cancer_detection():
    """Load the Cancer Detection tool"""
    try:
        # Add the Cancer Detection directory to path
        cancer_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "Disease Detection",
            "Cancer Detection"
        )
        
        if cancer_path not in sys.path:
            sys.path.append(cancer_path)
        
        # Import and run the Cancer Detection app
        try:
            from app import main as cancer_main
            st.markdown("## 🔬 Cancer Detection")
            st.markdown("Analyze medical data to detect potential cancer cases.")
            cancer_main()
        except ImportError as e:
            st.error(f"Failed to import Cancer Detection module: {str(e)}")
            st.info(f"Python path: {sys.path}")
        except Exception as e:
            st.error(f"Error loading Cancer Detection: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the Cancer Detection module is properly set up.")

def load_lung_cancer_prediction():
    """Load the Lung Cancer Prediction tool"""
    try:
        # Add the Lung Cancer Prediction directory to path
        lung_cancer_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "Disease Detection",
            "lung-cancer-prediction"
        )
        
        if lung_cancer_path not in sys.path:
            sys.path.append(lung_cancer_path)
        
        # Import and run the Lung Cancer Prediction app
        try:
            from app import main as lung_cancer_main
            st.markdown("## 🫁 Lung Cancer Risk Assessment")
            st.markdown("Assess the risk of lung cancer based on health and lifestyle factors.")
            lung_cancer_main()
        except ImportError as e:
            st.error(f"Failed to import Lung Cancer Prediction module: {str(e)}")
            st.info(f"Python path: {sys.path}")
        except Exception as e:
            st.error(f"Error loading Lung Cancer Prediction: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the Lung Cancer Prediction module is properly set up.")

def load_diabetes_prediction():
    """Load the Diabetes Prediction tool"""
    st.info("Diabetes Prediction tool is coming soon!")
    # TODO: Implement Diabetes prediction integration

def main():
    """Main application function"""
    # Main header
    st.markdown("<h1 class='main-header'>🩺 Disease Detection</h1>", unsafe_allow_html=True)
    
    if st.session_state.selected_tool is None:
        show_main_dashboard()
    else:
        load_tool(st.session_state.selected_tool)

def show_main_dashboard():
    """Display the main dashboard with available tools"""
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #424242;'>
            Access specialized AI tools for detecting various medical conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Available tools section
    st.markdown("### Available Tools")
    
    # First row of tools
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Brain Tumor Detection Card
        tool_card(
            "Brain Tumor Detection",
            "Detect brain tumors in MRI scans using YOLOv11",
            "🧠",
            "Open Tool",
            "brain_tumor_detection",
            available=True
        )
        
        # Diabetes Prediction Card
        tool_card(
            "Diabetes Prediction",
            "Predict the likelihood of diabetes based on health metrics",
            "🩸",
            "Coming Soon",
            "diabetes_prediction",
            available=False
        )
    
    with col2:
        # Alzheimer's Prediction Card
        tool_card(
            "Alzheimer's Prediction",
            "Predict Alzheimer's disease from brain scans",
            "🧓",
            "Coming Soon",
            "alzheimers_prediction",
            available=False
        )
        
        # Cancer Detection Card
        tool_card(
            "Cancer Detection",
            "Detect various types of cancer from medical images",
            "🦠",
            "Coming Soon",
            "cancer_detection",
            available=False
        )
    
    with col3:
        # Lung Cancer Prediction Card
        tool_card(
            "Lung Cancer Risk Assessment",
            "Assess lung cancer risk based on health and lifestyle factors",
            "🫁",
            "Open Tool",
            "lung_cancer_prediction",
            available=True
        )
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This Disease Detection application provides AI-powered tools for early detection of various medical conditions.
    The tools are designed to assist healthcare professionals and should not be used as a substitute for professional
    medical advice, diagnosis, or treatment.
    
    **Note:** Some tools may require specific model files or additional setup to function properly.
    """)

if __name__ == "__main__":
    main()