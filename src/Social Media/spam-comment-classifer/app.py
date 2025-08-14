import streamlit as st

st.set_page_config(page_title="Spam Comment Classifier", layout="centered")
st.title("🧹 Spam Comment Classifier")

comment = st.text_area("Paste a comment:")

if st.button("Detect"):
    if comment:
        if "buy now" in comment.lower() or "free" in comment.lower():
            st.error("⚠️ Spam Detected")
        else:
            st.success("✅ Not Spam")
    else:
        st.warning("Please enter a comment.")
