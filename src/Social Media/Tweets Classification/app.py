import streamlit as st

st.set_page_config(page_title="Tweets Classification", layout="centered")
st.title("Tweets Classification")

text = st.text_area("Enter a tweet:")

if st.button("Classify"):
    if text:
        st.info(f"Prediction: Positive" if "good" in text.lower() else "Prediction: Negative 👎")
    else:
        st.warning("Please enter a tweet to classify.")
