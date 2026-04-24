from fastapi import FastAPI
from app.schemas import CropInput
from app.model_loader import predict

app = FastAPI()

@app.get("/")
def home():
    return {"status": "API Running"}

@app.post("/predict")
def predict_crop(data: CropInput):
    values = [
        data.nitrogen,
        data.phosphorus,
        data.potassium,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]

    result = predict(values)

    return {"crop": result}