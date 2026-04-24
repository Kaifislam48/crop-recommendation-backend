import joblib
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

model = joblib.load(os.getenv("MODEL_PATH", "model/model.pkl"))
scaler = joblib.load(os.getenv("SCALER_PATH", "model/scaler.pkl"))
label_encoder = joblib.load(os.getenv("ENCODER_PATH", "model/label_encoder.pkl"))

def predict(data):
    input_data = np.array([[
        data.nitrogen,
        data.phosphorus,
        data.potassium,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Decode label
    result = label_encoder.inverse_transform(prediction)

    return result[0]