import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("student_model_1.pkl")

st.title("📘 Student Weak Topic Predictor")

# --- File Upload Section ---
st.header("Upload Student Scores (Excel)")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        st.subheader("Input Data:")
        st.write(df)

        # Make predictions
        predictions = model.predict(df[['Part A', 'Part B', 'Part C', 'Part D', 'Part E']])
        df['Predicted Weak Topic'] = predictions

        st.subheader("Prediction Results:")
        st.dataframe(df)

        # Download button
        st.download_button(
            label="📥 Download Results as Excel",
            data=df.to_excel(index=False, engine='openpyxl'),
            file_name="predicted_weak_topics.xlsx"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# --- Manual Entry Section ---
st.header("Or Manually Enter a Student's Scores")

with st.form("manual_input"):
    part_a = st.slider("Part A", 0, 5, 3)
    part_b = st.slider("Part B", 0, 5, 3)
    part_c = st.slider("Part C", 0, 5, 3)
    part_d = st.slider("Part D", 0, 5, 3)
    part_e = st.slider("Part E", 0, 5, 3)
    submit = st.form_submit_button("Predict")

    if submit:
        new_df = pd.DataFrame([{
            'Part A': part_a,
            'Part B': part_b,
            'Part C': part_c,
            'Part D': part_d,
            'Part E': part_e
        }])

        prediction = model.predict(new_df)[0]
        st.success(f"📌 Predicted Weakest Topic: {prediction}")
