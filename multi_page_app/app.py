import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Multi-Page ML App", layout="wide")

st.title("📊 Multi-Page Machine Learning App")

# Sidebar for uploading file
st.sidebar.title("📂 Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Save uploaded file in session_state
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.sidebar.success("✅ File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error reading the file: {e}")
else:
    st.sidebar.info("⬆️ Please upload a CSV file to continue.")

# Show preview of the dataset on main page
if "df" in st.session_state:
    st.subheader("🔍 Preview of Uploaded Dataset")
    st.dataframe(st.session_state["df"])
else:
    st.warning("⚠️ No dataset uploaded. Please upload a CSV file from the sidebar.")

# Add footer or description
st.markdown("---")
st.markdown("Navigate through pages (ETL, Model Training, Prediction, SHAP) using the left sidebar.")
