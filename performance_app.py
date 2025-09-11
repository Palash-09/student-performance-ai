# performance_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------------
# Load model (joblib/pickle)
# -------------------------------
def load_model(path="student_model_1.pkl"):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

@st.cache_resource
def get_model(path="student_model_1.pkl"):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found: {path}")
        return None
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# Expected feature columns (update if model trained differently)
FEATURE_COLS = ["Part A", "Part B", "Part C", "Part D", "Part E"]

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Student Performance Review",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel/CSV file", type=["xlsx", "xls", "csv"]
)
st.sidebar.markdown("---")
st.sidebar.info("Expected columns:\n- Roll Number\n- Student Name\n- Part A – Part E (scores 0–5)")

# -------------------------------
# Main Header
# -------------------------------
st.title("📊 AI-based Student Performance Review")
st.markdown(
    "This system predicts **individual weak topics** and identifies the "
    "**overall weakest topic in the class** using a trained ML model "
    "(`student_model_1.pkl`)."
)

# -------------------------------
# Load Model
# -------------------------------
model = get_model("student_model_1.pkl")

if uploaded_file is None:
    st.warning("⬆️ Please upload a student performance file to continue.")
    st.stop()

# -------------------------------
# Read uploaded data
# -------------------------------
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error("❌ Could not read file: " + str(e))
    st.stop()

st.subheader("📑 Uploaded Data Preview")
st.dataframe(df.head(), use_container_width=True)

# -------------------------------
# Ensure Roll Number & Student Name exist
# -------------------------------
if "Roll Number" not in df.columns or "Student Name" not in df.columns:
    st.error("❌ Uploaded file must have 'Roll Number' and 'Student Name' columns.")
    st.stop()

# -------------------------------
# Map feature columns
# -------------------------------
def map_feature_cols(df, expected):
    mapping = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for e in expected:
        if e in df.columns:
            mapping[e] = e
        elif e.lower() in cols_lower:
            mapping[e] = cols_lower[e.lower()]
        else:
            mapping[e] = None
    return mapping

col_map = map_feature_cols(df, FEATURE_COLS)
missing = [k for k, v in col_map.items() if v is None]
if missing:
    st.error(f"❌ Missing columns in uploaded file: {missing}")
    st.stop()

# Prepare features
X = pd.DataFrame()
for e in FEATURE_COLS:
    X[e] = pd.to_numeric(df[col_map[e]], errors="coerce").fillna(0)

# -------------------------------
# Run Predictions
# -------------------------------
if model is None:
    st.stop()

try:
    preds = model.predict(X)
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

df["Predicted Weak Topic"] = preds

# -------------------------------
# Show Results
# -------------------------------
st.header("👩‍🎓 Student Analysis")
st.dataframe(
    df[["Roll Number", "Student Name", *FEATURE_COLS, "Predicted Weak Topic"]],
    use_container_width=True
)

topic_counts = df["Predicted Weak Topic"].value_counts()
if not topic_counts.empty:
    overall_weak = topic_counts.idxmax()
    st.header("🏆 Class Insights")
    st.success(f"**Overall Weakest Topic:** {overall_weak} "
               f"({topic_counts.loc[overall_weak]} students)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Distribution of Weak Topics")
        fig1, ax1 = plt.subplots()
        ax1.bar(topic_counts.index.astype(str), topic_counts.values, color="#1f77b4")
        ax1.set_xlabel("Topic")
        ax1.set_ylabel("Number of Students")
        ax1.set_title("Weak Topics Across Class")
        st.pyplot(fig1)

    with col2:
        st.subheader("🧩 Proportion of Weak Topics")
        fig2, ax2 = plt.subplots()
        ax2.pie(topic_counts.values, labels=topic_counts.index.astype(str), autopct="%1.1f%%")
        ax2.set_title("Weak Topics Share")
        st.pyplot(fig2)

# -------------------------------
# Download Results
# -------------------------------
st.header("💾 Download Results")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Predictions as CSV",
    data=csv_bytes,
    file_name="predicted_weak_topics.csv",
    mime="text/csv"
)
