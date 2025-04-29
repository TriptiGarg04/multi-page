import streamlit as st
import pandas as pd

st.set_page_config(page_title="Multi-Page ML App", layout="wide")

st.sidebar.title("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df
    st.success("✅ Dataset uploaded successfully!")
else:
    st.info("📄 Please upload a dataset from the sidebar.")
