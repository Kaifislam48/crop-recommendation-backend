import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features (ONLY ONE SCALER)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y_encoded)

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save everything
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("✅ Model trained and saved successfully")