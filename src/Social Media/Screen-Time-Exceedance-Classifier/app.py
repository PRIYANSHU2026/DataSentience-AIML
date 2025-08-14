import streamlit as st

st.set_page_config(page_title="Screen Time Exceedance Classifier", layout="centered")
st.title("⏱️ Screen-Time Exceedance Classifier")

hours = st.slider("Enter daily screen time (in hours)", min_value=0.0, max_value=24.0, step=0.5)

if st.button("Classify"):
    if hours > 6:
        st.error("⚠️ Risk of Exceedance Detected")
    else:
        st.success("✅ Within Healthy Range")
