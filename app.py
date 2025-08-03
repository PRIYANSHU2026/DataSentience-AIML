import streamlit as st
import os
import sys
import importlib.util

# Configure page
st.set_page_config(
    page_title="DataSentience-AIML",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #2196F3;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .domain-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🌐 DataSentience-AIML</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI/ML Solutions Across Multiple Domains</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")

# Define domains and their modules
domains = {
    "Agricultural Solutions": [
        "Air Quality Index",
        "Crop Recomendation System",
        "Crop Recommendation System",
        "Crop-Recomendation-System-Using-Machine-Learning",
        "Crop Yield Prediction",
        "Decision tree regressor,ANN",
        "Fertiliser Recommendation System",
        "Plant Disease Detection",
        "Plant Seedlings Classification",
        "Soil Classifier CNN"
        "Tomato Disease Detection"
    ],
    "Environmental Monitoring": [
        "AQI Predictor lstm",
        "AquaGuard AI",
        "Bird Species Classification Web App",
        "Fire Detection",
        "Pollution-spike-alert",
        "Waste Classification",
        "Weather Prediction",
        "rainfall-prediction"
    ],
    "Healthcare": [
        "Disease Detection",
        "Doctor speciality Recommendation",
        "Emotion Recognition Based on NLP",
        "Health Monitoring",
        "heart failure",
        "Heart_Predection",
        "Medical Imaging",
        "medical-insurance",
        "misc",
        "Multilingual_Symptom_Checker"
    ],
    "Financial Analysis": [
        "Business Analytics",
        "Credit Card Eligibility Checker",
        "Cryptocurrency",
        "Market Analysis"
    ],
    "Entertainment Industry": [
        "RestaurantInsight AI",
        "SongGenreClassifier",
        "Youtube Comment Analysis",
        "movie_success_predictor"
    ],
    "Social Media": [
        "AI-trupet-midi-generator"
        "Hashtag Popularity Predictor",
        "Screen-Time-Exceedance-Classifier",
        "SentimentSense AI",
        "Toxic comment detector",
        "Tweets Classification",
        "spam-comment-classifer",
        "twitter-sentiment-analyser"
    ],
    "Machine Learning Techniques": [
        "Audio Classification",
        "AudioFlow Pro",
        "Bidirectional STM",
        "CBT_Chatbot",
        "Chatbot Using RASA",
        "Class Imbalance problem",
        "COVID_19-DATA-ANALYSIS",
        "Credit_card_fraud_detection",
        "Customer Segmentation using Machine Learning",
        "Email Classifier",
        "Exploratory-data-analysis",
        "Federated Learning",
        "Google Teachable Machine",
        "House Price Prediction (Linear vs Ridge vs Lasso)",
        "job_Advertisement_detection",
        "LanguageDetection",
        "Malicious Url Detection",
        "MNIST English Classification",
        "Nasa-Asteroid-Dataset-Analysis",
        "OCR Systems",
        "Random forest test",
        "Social Media Fake Accounts Detection with Interactive UI",
        "Spelling Corrector",
        "Website Classification",
        "spam-detection-model"
        "Sudoku Solver using CNN",
        "Website Forecasting Using ARIMA"
    ],
    "Time series and price forecasting": [
        "Time series Forecasting with Python",
        "Time-Series LSTM Model",
        "Titanic-Survival-Prediction"
    ],
    "Transportation & Safety": [
        "Driver Safety",
        "Traffic Management"
    ]
}

# Domain selection
selected_domain = st.sidebar.selectbox("Select Domain", list(domains.keys()))

# Module selection based on domain
selected_module = st.sidebar.selectbox("Select Module", domains[selected_domain])

# About section in sidebar
with st.sidebar.expander("About"):
    st.write("""
    This application integrates various AI/ML solutions developed as part of the Social Summer of Code 2025 & GirlScript Summer of Code 2025 initiatives.
    
    The project covers multiple domains including Healthcare, Finance, Agriculture, NLP, Safety, and more.
    
    Select a domain and a specific module from the dropdowns above to explore different AI/ML solutions.
    """)

# Main content area
st.markdown(f'<div class="domain-title">{selected_domain} » {selected_module}</div>', unsafe_allow_html=True)

# Function to load and run module
def load_module(module_path, module_name):
    try:
        # Add module directory to path
        module_dir = os.path.dirname(module_path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
            
        # Import the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            return False
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        st.error(f"Error loading module: {str(e)}")
        return False

# Try to load the selected module
module_path = os.path.join("src", selected_domain, selected_module, "app.py")
if os.path.exists(module_path):
    success = load_module(module_path, selected_module.replace(" ", "_"))
    if not success:
        st.warning(f"The module '{selected_module}' could not be loaded directly.")
        st.info("Here's a description of what this module does:")
        
        # Display module description
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Try to load README for the module
            readme_path = os.path.join("src", selected_domain, selected_module, "README.md")
            if os.path.exists(readme_path):
                with open(readme_path, "r") as f:
                    readme_content = f.read()
                st.markdown(readme_content)
            else:
                st.write(f"This module provides AI/ML solutions for {selected_module}.")
                st.write("Detailed documentation is not available for this module.")
            
            # Show path to access the module directly
            st.code(f"cd {os.path.join('src', selected_domain, selected_module)}\nstreamlit run app.py")
            
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning(f"The module '{selected_module}' does not have a Streamlit app.py file.")
    
    # Display module description
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Try to load README for the module
        readme_path = os.path.join("src", selected_domain, selected_module, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                readme_content = f.read()
            st.markdown(readme_content)
        else:
            st.write(f"This module provides AI/ML solutions for {selected_module}.")
            st.write("Detailed documentation is not available for this module.")
        
        # List files in the module directory
        module_dir = os.path.join("src", selected_domain, selected_module)
        if os.path.exists(module_dir):
            files = os.listdir(module_dir)
            if files:
                st.write("Files in this module:")
                for file in files:
                    if file.endswith(".py") or file.endswith(".ipynb"):
                        st.code(file)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 DataSentience-AIML | Social Summer of Code 2025 & GirlScript Summer of Code 2025</div>', unsafe_allow_html=True)