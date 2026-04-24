import joblib
import os
from dotenv import load_dotenv

load_dotenv()

model = joblib.load(os.getenv("MODEL_PATH"))
minmax = joblib.load(os.getenv("MINMAX_PATH"))
stand = joblib.load(os.getenv("STAND_PATH"))

# 👇 ADD THIS
label_encoder = joblib.load("model/label_encoder.pkl")

def predict(data):
    data = [data]
    data = minmax.transform(data)
    data = stand.transform(data)

    result = model.predict(data)

    # 👇 Convert number → crop name
    crop_name = label_encoder.inverse_transform(result)

    return crop_name[0]