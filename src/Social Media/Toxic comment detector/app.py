import streamlit as st

st.set_page_config(page_title="Toxic Comment Detector", layout="centered")
st.title("💬 Toxic Comment Detector")

text = st.text_area("Enter a comment:")

if st.button("Analyze"):
    if text:
        st.warning("⚠️ Toxic content detected." if "hate" in text.lower() else "✅ Non-toxic comment.")
    else:
        st.warning("Please enter some content.")
