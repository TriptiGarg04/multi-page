import streamlit as st
import shap
import matplotlib.pyplot as plt

st.title("🔍 SHAP Explainability")

# Check if dataset and model exist
if "df" not in st.session_state or "model" not in st.session_state:
    st.warning("⚠️ Please upload a dataset and train a model first.")
    st.stop()

df = st.session_state["df"]
model = st.session_state["model"]

# Optional: choose feature columns
X = df.select_dtypes(include=['float64', 'int64']).dropna()

# Explain model predictions using SHAP
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)

st.subheader("SHAP Summary Plot")
fig = plt.figure()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)
