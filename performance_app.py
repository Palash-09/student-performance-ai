# performance_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------------
# Load model (try joblib, fallback to pickle)
# -------------------------------
def load_model(path="student_model_1.pkl"):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

@st.cache(allow_output_mutation=True)
def get_model(path="student_model_1.pkl"):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found: {path}")
        return None
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# Expected feature columns (update if your model was trained with different ones)
FEATURE_COLS = ["Part A", "Part B", "Part C", "Part D", "Part E"]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Student Performance Review", layout="wide")
st.title("📊 AI-based Student Performance Review")
st.markdown("""
Upload an Excel/CSV file with columns:  
**Student ID, Part A, Part B, Part C, Part D, Part E** (scores 0–5).  

The app uses **student_model_1.pkl** to predict each student's weakest topic,  
and also shows the **overall weakest topic in the class**.
""")

# Load trained model
model = get_model("student_model_1.pkl")

uploaded_file = st.file_uploader("📂 Upload Excel/CSV file", type=["xlsx", "xls", "csv"])

if uploaded_file is None:
    st.info("⬆️ Please upload a file to start analysis.")
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
# Ensure Student ID column
# -------------------------------
possible_id_cols = ["Student ID", "ID", "Roll No", "RollNumber"]
id_col = None
for c in possible_id_cols:
    if c in df.columns:
        id_col = c
        break
if id_col is None:
    df.insert(0, "Student ID", range(1, len(df) + 1))
    id_col = "Student ID"

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
st.subheader("👩‍🎓 Student-wise Predictions")
st.dataframe(df[[id_col, *FEATURE_COLS, "Predicted Weak Topic"]], use_container_width=True)

# Overall weakest topic
topic_counts = df["Predicted Weak Topic"].value_counts()
if not topic_counts.empty:
    overall_weak = topic_counts.idxmax()
    st.subheader("🏆 Overall Weakest Topic in Class")
    st.success(f"{overall_weak} — {topic_counts.loc[overall_weak]} students")

    # Bar Chart
    st.subheader("📊 Weak Topic Distribution")
    fig1, ax1 = plt.subplots()
    ax1.bar(topic_counts.index.astype(str), topic_counts.values)
    ax1.set_xlabel("Topic")
    ax1.set_ylabel("Number of Students")
    ax1.set_title("Predicted Weak Topics Across Class")
    st.pyplot(fig1)

    # Pie Chart
    st.subheader("🧩 Weak Topic Proportions")
    fig2, ax2 = plt.subplots()
    ax2.pie(topic_counts.values, labels=topic_counts.index.astype(str), autopct="%1.1f%%")
    ax2.set_title("Weak Topics Share")
    st.pyplot(fig2)

# -------------------------------
# Download Predictions
# -------------------------------
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download Predictions (CSV)", data=csv_bytes,
                   file_name="predicted_weak_topics.csv", mime="text/csv")
