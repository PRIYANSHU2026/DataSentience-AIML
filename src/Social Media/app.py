import streamlit as st
import os

st.set_page_config(page_title="Social Media Projects", layout="centered")
st.title("📱 Social Media AI Tools")
st.markdown("""
Explore intelligent systems built to understand, classify, and analyze user behavior and content in social media platforms.
These projects tackle NLP, moderation, music generation, and trend analysis.
""")

# --- Project Listing ---
base_path = "src/Social Media"
projects = [
    "AI-trumpet-midi-generator",
    "Hashtag Popularity Predictor",
    "Screen-Time-Exceedance-Classifier",
    "SentimentSense AI",
    "spam-comment-classifier",
    "Toxic comment detector",
    "Tweets Classification",
    "twitter-sentiment-analyser"
]

st.subheader("Available Modules")

for project in projects:
    display_name = project.replace("-", " ").replace("_", " ").title()
    app_path = f"{base_path}/{project}/app.py"

    st.markdown(f"### 🔹 {display_name}")
    st.markdown(f"Location: `{app_path}`")

    st.markdown(f"""
    <a href="/{app_path}" target="_self">
        <button style="background-color:#1f77b4; color:white; padding:6px 16px; border:none; border-radius:6px; margin-bottom:10px;">
            ▶️ Open Interface
        </button>
    </a>
    """, unsafe_allow_html=True)

    st.markdown("---")
