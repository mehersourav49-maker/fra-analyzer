import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load Training Data (simulated FRA)
# -------------------------------
df = pd.read_csv("fra_simulated_data.csv")

# Prepare dataset
X = []
y = []

for val in df["Healthy_Response"]:
    X.append([val])
    y.append(0)  # Healthy

for val in df["Faulty_Response"]:
    X.append([val])
    y.append(1)  # Faulty

X = np.array(X)
y = np.array(y)

# Train model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("⚡ AI-Powered FRA Analyzer for Transformers")
st.write("Upload FRA CSV data and let the AI classify Healthy vs Faulty transformer signatures.")

uploaded_file = st.file_uploader("Upload FRA CSV file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded CSV
    user_df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.write(user_df.head())

    # Check columns
    if "Frequency_Hz" in user_df.columns and ("Response" in user_df.columns or "Healthy_Response" in user_df.columns or "Faulty_Response" in user_df.columns):
        
        # Pick response column automatically
        response_col = None
        for col in ["Response", "Healthy_Response", "Faulty_Response"]:
            if col in user_df.columns:
                response_col = col
                break
        
        if response_col:
            responses = user_df[response_col].values.reshape(-1,1)
            predictions = model.predict(responses)

            # Count results
            healthy_count = np.sum(predictions == 0)
            faulty_count = np.sum(predictions == 1)

            st.subheader("🔍 Prediction Results")
            st.write(f"✅ Healthy Points: {healthy_count}")
            st.write(f"⚠️ Faulty Points: {faulty_count}")

            if faulty_count > healthy_count:
                st.error("Transformer Condition: **FAULTY**")
            else:
                st.success("Transformer Condition: **HEALTHY**")

            # Plot
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(user_df["Frequency_Hz"], user_df[response_col], label="FRA Signature", color="blue")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Response")
            ax.set_title("FRA Curve")
            ax.grid(True)
            st.pyplot(fig)
    else:
        st.warning("CSV must contain 'Frequency_Hz' and a response column.")
import joblib

# Load the pre-trained model
model = joblib.load("model.pkl")
