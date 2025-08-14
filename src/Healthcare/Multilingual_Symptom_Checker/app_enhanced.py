"""
Enhanced Multilingual Symptom Checker - Streamlit App
"""
import streamlit as st
from chatbot_utils import translate_to_english, translate_from_english, get_response_from_gemini, clean_ai_response
from config import configure_gemini
import time

# Set page config
st.set_page_config(
    page_title="Multilingual Symptom Checker",
    page_icon="🏥",
    layout="centered"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        color: #1e88e5;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .symptom-input {
        margin-bottom: 1.5rem;
    }
    .result-box {
        background-color: #f5f9ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1976d2;
    }
    .urgency-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .urgency-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .urgency-low {
        color: #388e3c;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .language-selector {
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Available languages
LANGUAGES = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "বাংলা (Bengali)": "bn",
    "தமிழ் (Tamil)": "ta",
    "తెలుగు (Telugu)": "te",
    "मराठी (Marathi)": "mr",
    "ગુજરાતી (Gujarati)": "gu",
    "ਪੰਜਾਬੀ (Punjabi)": "pa",
    "Español (Spanish)": "es",
    "Français (French)": "fr",
    "Deutsch (German)": "de",
    "中文 (Chinese)": "zh",
    "العربية (Arabic)": "ar"
}

def get_urgency_class(urgency):
    """Get CSS class based on urgency level"""
    urgency = str(urgency).lower()
    if 'emergency' in urgency:
        return 'urgency-high'
    elif 'moderate' in urgency:
        return 'urgency-medium'
    else:
        return 'urgency-low'

def format_response(response):
    """Format the AI response into a structured output"""
    lines = response.split('\n')
    formatted = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.lower().startswith('possible cause'):
            formatted += f"### 🔍 {line}\n\n"
        elif line.lower().startswith('urgency level'):
            urgency_class = get_urgency_class(line)
            formatted += f"### ⚠️ Urgency Level\n"
            formatted += f"<div class='{urgency_class}'>"
            # Extract just the urgency level (e.g., "High") for display
            urgency_text = line.split(':', 1)[1].strip() if ':' in line else line
            formatted += f"{urgency_text}"
            formatted += "</div>\n\n"
        elif line.lower().startswith('suggestion'):
            formatted += f"### 💡 Suggestion\n"
            # Extract the suggestion text
            suggestion = line.split(':', 1)[1].strip() if ':' in line else ""
            formatted += f"{suggestion}\n\n"
        elif 'note:' in line.lower() or 'disclaimer' in line.lower():
            formatted += f"---\n\n> ℹ️ {line}\n"
        else:
            formatted += f"{line}\n"
    
    return formatted

def main():
    """Main function to run the Streamlit app"""
    st.markdown('<h1 class="main-header">🌍 Multilingual Symptom Checker</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; color: #424242;'>
            Describe your symptoms in your preferred language and get medical guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Language selection
    selected_lang = st.selectbox(
        "🌐 Select your preferred language:",
        list(LANGUAGES.keys()),
        index=0,
        key="lang_select",
        help="Choose the language you're most comfortable with"
    )
    
    # Symptom input
    st.markdown("### 🩺 Describe Your Symptoms")
    user_input = st.text_area(
        "",
        placeholder="Example: I have a severe headache and fever since yesterday...",
        height=150,
        key="symptom_input"
    )
    
    # Check symptoms button
    col1, col2 = st.columns([1, 3])
    with col1:
        check_button = st.button("🔍 Check Symptoms", use_container_width=True)
    
    # Add some space
    st.markdown("")
    
    # Initialize session state for response if it doesn't exist
    if 'last_response' not in st.session_state:
        st.session_state.last_response = None
    
    # Process input when button is clicked
    if check_button:
        if user_input.strip():
            with st.spinner("Analyzing your symptoms..."):
                try:
                    # Show loading animation
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.02)  # Simulate processing time
                        progress_bar.progress(percent_complete + 1)
                    
                    # Get response from Gemini
                    translated_input = translate_to_english(user_input, LANGUAGES[selected_lang])
                    llm_response = get_response_from_gemini(translated_input, configure_gemini())
                    
                    # Translate response back to user's language
                    translated_output = translate_from_english(
                        clean_ai_response(llm_response), 
                        LANGUAGES[selected_lang]
                    )
                    
                    # Store response in session state
                    st.session_state.last_response = translated_output
                    
                    # Clear the progress bar
                    progress_bar.empty()
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please try again or check your internet connection.")
        else:
            st.warning("Please describe your symptoms before checking.")
    
    # Display the last response if available
    if st.session_state.last_response:
        st.markdown("## 📋 Analysis Results")
        st.markdown("<div class='result-box'>" + format_response(st.session_state.last_response) + "</div>", 
                   unsafe_allow_html=True)
        
        # Add a button to start a new check
        if st.button("🔄 Start a New Symptom Check"):
            st.session_state.last_response = None
            st.rerun()
    
    # Add disclaimer
    st.markdown("---")
    st.markdown("""
    ### ℹ️ Important Note
    This tool provides general health information and is not a substitute for professional medical advice, 
    diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider 
    with any questions you may have regarding a medical condition.
    """)

if __name__ == "__main__":
    main()
