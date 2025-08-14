import streamlit as st

st.set_page_config(page_title="SentimentSense AI", layout="centered")
st.title("🧠 SentimentSense AI")

text = st.text_area("Analyze Sentiment:")

if st.button("Analyze"):
    if text:
        st.info(f"Sentiment: {'😊 Positive' if 'happy' in text.lower() else '😠 Negative'} [Mock Output]")
    else:
        st.warning("Enter some text to analyze.")
