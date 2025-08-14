import streamlit as st

st.set_page_config(page_title="Hashtag Popularity Predictor", layout="centered")
st.title("🏷️ Hashtag Popularity Predictor")

hashtag = st.text_input("Enter a hashtag (without #):")

if st.button("Predict"):
    if hashtag:
        st.success(f"🔥 #{hashtag} is trending! Popularity Score: 87% [Simulated]")
    else:
        st.warning("Please enter a hashtag.")
