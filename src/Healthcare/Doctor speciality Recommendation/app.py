"""
Doctor Specialty Recommendation - Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Set page config
st.set_page_config(
    page_title="Doctor Specialty Recommendation",
    page_icon="👨‍⚕️",
    layout="centered"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1e88e5;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .specialist-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1976d2;
    }
    .disease-info {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the required data files"""
    base_path = Path(__file__).parent
    
    # Load disease descriptions
    disease_desc = pd.read_csv(base_path / 'Disease_Description.csv')
    
    # Load doctor specialists
    specialists = pd.read_csv(base_path / 'Doctor_Specialist.csv')
    
    # Create a mapping of diseases to specialists (simplified for this example)
    # In a real implementation, you would load this from Doctor_Versus_Disease.csv
    disease_specialist = {
        'Drug Reaction': 'Dermatologist',
        'Malaria': 'Infectious Disease Specialist',
        'Allergy': 'Allergist',
        'Hypothyroidism': 'Endocrinologist',
        'Chicken pox': 'Pediatrician',
        'Cervical spondylosis': 'Neurologist',
        'Hyperthyroidism': 'Endocrinologist',
        'Urinary tract infection': 'Urologist',
        'Varicose veins': 'Phlebologist',
        'Hepatitis D': 'Hepatologist',
        'Heart attack': 'Cardiologist',
        'Pneumonia': 'Pulmonologist',
        'Arthritis': 'Rheumatologist',
        'Gastroenteritis': 'Gastroenterologist',
    }
    
    return disease_desc, specialists, disease_specialist

def get_specialist_recommendation(disease, disease_desc, disease_specialist):
    """Get specialist recommendation for a given disease"""
    # Find disease description
    description = disease_desc[disease_desc['Disease'].str.lower() == disease.lower()]
    
    if len(description) == 0:
        return None, None, "Disease not found in our database."
    
    disease_name = description.iloc[0]['Disease']
    disease_desc = description.iloc[0]['Description']
    
    # Get recommended specialist
    specialist = disease_specialist.get(disease_name, 'General Physician')
    
    return disease_name, specialist, disease_desc

def main():
    """Main function to run the Streamlit app"""
    st.markdown('<h1 class="main-header">👨‍⚕️ Doctor Specialty Recommendation</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    disease_desc, specialists, disease_specialist = load_data()
    
    # Get list of diseases for the dropdown
    disease_list = disease_desc['Disease'].sort_values().tolist()
    
    # Add search functionality
    st.markdown("### Find the Right Specialist")
    st.markdown("Enter your symptoms or select a condition to find the appropriate medical specialist.")
    
    # Search box for disease
    selected_disease = st.selectbox(
        "Select or type a medical condition:",
        [""] + disease_list,
        format_func=lambda x: "Select a condition..." if x == "" else x
    )
    
    if selected_disease:
        # Get recommendation
        disease_name, specialist, description = get_specialist_recommendation(
            selected_disease, disease_desc, disease_specialist
        )
        
        if disease_name:
            # Display disease information
            st.markdown(f"### {disease_name}")
            st.markdown(f"<div class='disease-info'>{description}</div>", unsafe_allow_html=True)
            
            # Display specialist recommendation
            st.markdown("### Recommended Specialist")
            st.markdown(f"""
            <div class='specialist-card'>
                <h3>👨‍⚕️ {specialist}</h3>
                <p>A {specialist.lower()} specializes in treating conditions like {disease_name.lower()}.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional information
            st.markdown("### Next Steps")
            st.markdown("""
            1. **Schedule an appointment** with a {0}
            2. **Prepare for your visit** by noting down your symptoms and any questions
            3. **Bring relevant medical records** to your appointment
            4. **Follow up** as recommended by your healthcare provider
            """.format(specialist.lower()))
            
            # Find nearby doctors (placeholder functionality)
            if st.button("🔍 Find {0} Near Me".format(specialist)):
                st.info("This feature would connect to a doctor directory service in a production environment.")
        else:
            st.warning("We couldn't find a specific specialist for this condition. Please consult with a general physician.")
    
    # Add information about different specialists
    with st.expander("ℹ️ Common Medical Specialists"):
        st.markdown("""
        - **Cardiologist**: Heart and blood vessel conditions
        - **Dermatologist**: Skin, hair, and nail conditions
        - **Endocrinologist**: Hormonal and metabolic disorders
        - **Gastroenterologist**: Digestive system disorders
        - **Neurologist**: Brain and nervous system disorders
        - **Pulmonologist**: Lung and respiratory conditions
        - **Rheumatologist**: Joint and musculoskeletal disorders
        - **Urologist**: Urinary tract and male reproductive system
        """)
    
    # Add disclaimer
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This tool provides general information and does not replace professional medical advice. 
    Always consult with a qualified healthcare provider for diagnosis and treatment.
    """)

if __name__ == "__main__":
    main()
