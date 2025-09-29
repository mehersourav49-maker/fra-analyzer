import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------
# Load the simulated FRA data
# -------------------------------
df = pd.read_csv("fra_simulated_data.csv")

# Prepare dataset: combine Healthy and Faulty into one column
X = []
y = []

# Healthy data
for val in df["Healthy_Response"]:
    X.append([val])
    y.append(0)  # 0 = Healthy

# Faulty data
for val in df["Faulty_Response"]:
    X.append([val])
    y.append(1)  # 1 = Faulty

X = np.array(X)
y = np.array(y)

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Train Random Forest Classifier
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

print("✅ Model Training Complete!\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Healthy", "Faulty"]))

# -------------------------------
# Quick Test (predict new point)
# -------------------------------
sample_val = [[0.8]]  # example response amplitude
prediction = model.predict(sample_val)
print("\nPrediction for sample value 0.8 →", "Faulty" if prediction[0]==1 else "Healthy")

# -------------------------------
# Plotting FRA signatures (with labels)
# -------------------------------
plt.figure(figsize=(10,5))
plt.scatter(range(len(df["Healthy_Response"])), df["Healthy_Response"], label="Healthy", alpha=0.6, color="green")
plt.scatter(range(len(df["Faulty_Response"])), df["Faulty_Response"], label="Faulty", alpha=0.6, color="red")
plt.title("FRA Training Data: Healthy vs Faulty")
plt.xlabel("Sample Index")
plt.ylabel("Response Value")
plt.legend()
plt.grid(True)
plt.show()
import joblib

# Save trained model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
