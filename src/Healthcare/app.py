"""
Healthcare - Main Application
"""
import streamlit as st
from pathlib import Path
import os
import sys

# Set page config
st.set_page_config(
    page_title="Healthcare AI",
    page_icon="🏥",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0d47a1;
        font-weight: 500;
        margin: 1rem 0;
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

def tool_card(title, description, icon, button_text, key):
    """Create a tool card component"""
    with st.container():
        st.markdown(f"""
        <div class="tool-card">
            <h3>{icon} {title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(button_text, key=key, use_container_width=True):
            st.session_state.selected_tool = key
            st.rerun()

def go_back_to_main():
    """Reset to main dashboard"""
    st.session_state.selected_tool = None
    st.rerun()

def main():
    """Main application function"""
    # Main header
    st.markdown("<h1 class='main-header'>🏥 Healthcare AI Dashboard</h1>", unsafe_allow_html=True)
    
    if st.session_state.selected_tool is None:
        show_main_dashboard()
    else:
        load_tool(st.session_state.selected_tool)

def show_main_dashboard():
    """Display the main dashboard with available tools"""
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #424242;'>
            Access advanced healthcare AI tools for disease detection and medical analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tools section
    st.markdown("### Disease Detection Tools")
    st.markdown("Specialized tools for detecting various medical conditions.")
    
    # First row of tools
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Brain Tumor Detection Card
        tool_card(
            "Brain Tumor Detection",
            "Detect brain tumors in MRI scans using YOLOv11",
            "🧠",
            "Open Tool",
            "brain_tumor_detection"
        )
    
    with col2:
        # Cancer Detection Card
        tool_card(
            "Cancer Detection",
            "Analyze medical data to detect potential cancer cases",
            "🔬",
            "Open Tool",
            "cancer_detection"
        )
    
    with col3:
        # Doctor Specialty Recommendation Card
        tool_card(
            "Doctor Recommendation",
            "Find the right specialist for your medical condition",
            "👨‍⚕️",
            "Find Specialist",
            "doctor_recommendation"
        )
    
    # Second row of tools
    st.markdown("### Symptom Analysis Tools")
    st.markdown("Tools to help you understand and analyze symptoms.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Multilingual Symptom Checker Card
        tool_card(
            "Multilingual Symptom Checker",
            "Describe your symptoms in your preferred language and get guidance",
            "🌍",
            "Check Symptoms",
            "symptom_checker"
        )
    
    with col2:
        # Federated Learning Card
        tool_card(
            "Federated Learning",
            "Train ML models across devices while keeping data localized",
            "🤖",
            "Explore",
            "federated_learning"
        )
    
    with col3:
        # Placeholder for future tools
        tool_card(
            "Coming Soon",
            "More healthcare tools coming in future updates",
            "🔜",
            "Learn More",
            "coming_soon_4"
        )
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This healthcare dashboard provides access to AI-powered medical tools designed to assist healthcare professionals 
    in disease detection and analysis. The tools are for informational purposes only and should not be used as 
    a substitute for professional medical advice, diagnosis, or treatment.
    """)

def load_tool(tool_name):
    """Load and run the selected tool"""
    st.markdown("---")
    
    # Back button
    if st.button("← Back to Dashboard", key=f"back_{tool_name}"):
        go_back_to_main()
        return
    
    # Tool content area
    if tool_name == "brain_tumor_detection":
        load_brain_tumor_detection()
    elif tool_name == "cancer_detection":
        load_cancer_detection()
    elif tool_name == "doctor_recommendation":
        load_doctor_recommendation()
    elif tool_name == "symptom_checker":
        load_symptom_checker()
    elif tool_name == "federated_learning":
        load_federated_learning()
    else:
        st.warning("This tool is not yet available.")
        st.button("← Back to Dashboard", key=f"back_from_{tool_name}", on_click=go_back_to_main)

def load_brain_tumor_detection():
    """Load the Brain Tumor Detection tool"""
    try:
        # Add the Brain Tumor Detection directory to path
        brain_tumor_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "Disease Detection", 
            "Brain_Tumor_Detection_Using_YOLOv11"
        )
        
        if brain_tumor_path not in sys.path:
            sys.path.append(brain_tumor_path)
        
        # Import and run the Brain Tumor Detection app
        try:
            from app import main as brain_tumor_main
            brain_tumor_main()
        except ImportError as e:
            st.error(f"Failed to import Brain Tumor Detection module: {str(e)}")
            st.info(f"Python path: {sys.path}")
        except Exception as e:
            st.error(f"Error loading Brain Tumor Detection: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the Brain Tumor Detection module is properly set up.")

def load_cancer_detection():
    """Load the Cancer Detection tool from the Disease Detection module"""
    try:
        # Add the Disease Detection directory to path
        disease_detection_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "Disease Detection"
        )
        
        if disease_detection_path not in sys.path:
            sys.path.append(disease_detection_path)
        
        # Import and run the Disease Detection app
        try:
            from app import load_cancer_detection as disease_cancer_detection
            disease_cancer_detection()
        except ImportError as e:
            st.error(f"Failed to import Cancer Detection module: {str(e)}")
            st.info(f"Python path: {sys.path}")
        except Exception as e:
            st.error(f"Error loading Cancer Detection: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the Cancer Detection module is properly set up.")

def load_doctor_recommendation():
    """Load the Doctor Specialty Recommendation tool"""
    try:
        # Add the Doctor Specialty Recommendation directory to path
        doctor_rec_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "Doctor speciality Recommendation"
        )
        
        if doctor_rec_path not in sys.path:
            sys.path.append(doctor_rec_path)
        
        # Import and run the Doctor Recommendation app
        try:
            from app import main as doctor_rec_main
            st.markdown("## 👨‍⚕️ Doctor Specialty Recommendation")
            st.markdown("Find the right medical specialist for your condition.")
            doctor_rec_main()
        except ImportError as e:
            st.error(f"Failed to import Doctor Recommendation module: {str(e)}")
            st.info(f"Python path: {sys.path}")
        except Exception as e:
            st.error(f"Error loading Doctor Recommendation: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the Doctor Recommendation module is properly set up.")

def load_symptom_checker():
    """Load the Multilingual Symptom Checker tool"""
    try:
        # Add the Multilingual Symptom Checker directory to path
        symptom_checker_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "Multilingual_Symptom_Checker"
        )
        
        if symptom_checker_path not in sys.path:
            sys.path.append(symptom_checker_path)
        
        # Import and run the enhanced Symptom Checker app
        try:
            from app_enhanced import main as symptom_checker_main
            st.markdown("## 🌍 Multilingual Symptom Checker")
            st.markdown("Describe your symptoms in your preferred language and get medical guidance.")
            symptom_checker_main()
        except ImportError as e:
            st.error(f"Failed to import Symptom Checker module: {str(e)}")
            st.info(f"Python path: {sys.path}")
            # Fallback to the original app.py if enhanced version fails
            try:
                from app import main as symptom_checker_main
                st.warning("Using basic version of Symptom Checker. Some features may be limited.")
                symptom_checker_main()
            except Exception as e2:
                st.error(f"Error loading Symptom Checker: {str(e2)}")
        except Exception as e:
            st.error(f"Error in Symptom Checker: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the Symptom Checker module is properly set up.")

def load_federated_learning():
    """Load the Federated Learning tool"""
    try:
        # Add the Federated Learning directory to path
        federated_learning_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "Machine Learning Techniques",
            "Federated_Learning"
        )
        
        if federated_learning_path not in sys.path:
            sys.path.append(federated_learning_path)
        
        # Import and run the Federated Learning app
        try:
            from app import main as federated_learning_main
            st.markdown("## 🤖 Federated Learning")
            st.markdown("Train machine learning models across decentralized devices while keeping data localized.")
            federated_learning_main()
        except ImportError as e:
            st.error(f"Failed to import Federated Learning module: {str(e)}")
            st.info(f"Python path: {sys.path}")
        except Exception as e:
            st.error(f"Error in Federated Learning: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the Federated Learning module is properly set up.")

if __name__ == "__main__":
    main()
