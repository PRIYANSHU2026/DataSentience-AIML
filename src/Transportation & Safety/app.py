import streamlit as st
import os
import sys

# Set page config
st.set_page_config(
    page_title="Transportation & Safety Systems",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3498DB;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 500;
        color: #2E86C1;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #2C3E50;
    }
    .card-text {
        color: #555;
    }
    .btn-primary {
        background-color: #3498DB;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        margin-top: 1rem;
        display: inline-block;
        text-decoration: none;
        font-weight: 500;
    }
    .btn-primary:hover {
        background-color: #2980B9;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🚗 Transportation & Safety Systems</div>', unsafe_allow_html=True)
st.markdown("""This module provides AI/ML solutions for transportation safety and traffic management, 
including driver monitoring systems and intelligent traffic analysis.""")

# Main content
st.markdown('<div class="sub-header">Available Systems</div>', unsafe_allow_html=True)

# Create two columns for the cards
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🚗 Driver Safety</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-text">', unsafe_allow_html=True)
    st.write("""
    Systems to monitor and enhance driver safety:
    
    - **Drowsiness Detection**: Monitors driver alertness by analyzing eye patterns
    - **CCTV Accident Analysis**: Detects traffic accidents from surveillance footage
    
    These systems help prevent accidents caused by driver fatigue and enable rapid response to incidents.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Open Driver Safety Systems", key="driver_safety"):
        # This would normally redirect, but in Streamlit we'll use session state
        st.session_state.selected_module = "Driver Safety"
        st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🚦 Traffic Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-text">', unsafe_allow_html=True)
    st.write("""
    Systems to optimize traffic flow and safety:
    
    - **Traffic Sign Recognition**: Identifies and classifies traffic signs for driver assistance
    - **Ride Request Analysis**: Analyzes patterns in ride-sharing requests for demand prediction
    
    These systems enhance road safety and improve traffic efficiency through intelligent monitoring and analysis.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Open Traffic Management Systems", key="traffic_mgmt"):
        # This would normally redirect, but in Streamlit we'll use session state
        st.session_state.selected_module = "Traffic Management"
        st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Handle module selection if set
if 'selected_module' in st.session_state:
    if st.session_state.selected_module == "Driver Safety":
        # In a real app, we might import and run the module
        # For demo, we'll just show a message
        st.info("Loading Driver Safety module... In a production environment, this would load the Driver Safety app.")
        
        # Show command to run directly
        st.code("cd src/Transportation & Safety/Driver Safety\nstreamlit run app.py")
        
        # Clear selection after showing
        st.session_state.pop('selected_module')
    
    elif st.session_state.selected_module == "Traffic Management":
        # In a real app, we might import and run the module
        # For demo, we'll just show a message
        st.info("Loading Traffic Management module... In a production environment, this would load the Traffic Management app.")
        
        # Show command to run directly
        st.code("cd src/Transportation & Safety/Traffic Management\nstreamlit run app.py")
        
        # Clear selection after showing
        st.session_state.pop('selected_module')

# Additional information
st.markdown("### Integration with Main Application")
st.write("""
These Transportation & Safety systems are fully integrated with the main DataSentience-AIML application. 
You can access them through the main navigation by selecting 'Transportation & Safety' from the domain dropdown.

For direct access to specific modules, use the buttons above or navigate through the main application interface.
""")