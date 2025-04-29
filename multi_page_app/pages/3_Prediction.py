import streamlit as st
import pandas as pd

st.title("📈 Predictions")

if "model" not in st.session_state or "df" not in st.session_state:
    st.warning("⚠️ Train a model first in the Model Training page.")
    st.stop()

model = st.session_state["model"]
features = st.session_state.get("features")

# Upload new data for prediction
uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    
    if not all(f in input_df.columns for f in features):
        st.error(f"❌ Uploaded file must include these features: {features}")
    else:
        st.subheader("🔮 Prediction Results")
        predictions = model.predict(input_df[features])
        input_df["Prediction"] = predictions
        st.dataframe(input_df)

        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
