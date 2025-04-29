import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("⚙️ Model Training")

if "df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset in the ETL page first.")
    st.stop()

df = st.session_state["df"]

# Select features and target
columns = df.columns.tolist()
target = st.selectbox("Select Target Column", columns)

features = st.multiselect("Select Feature Columns", [col for col in columns if col != target])

if not features:
    st.info("Please select features to proceed.")
    st.stop()

X = df[features]
y = df[target]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"✅ Model trained with accuracy: {acc:.2f}")

# Store model and test data in session_state
st.session_state["model"] = model
st.session_state["X_test"] = X_test
st.session_state["y_test"] = y_test
st.session_state["features"] = features
