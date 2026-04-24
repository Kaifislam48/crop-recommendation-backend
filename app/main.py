from fastapi import FastAPI
from app.schemas import CropInput
from app.model_loader import predict

app = FastAPI()

@app.get("/")
def home():
    return {"status": "API Running"}

@app.post("/predict")
def predict_crop(data: CropInput):
    result = predict(data)

    return {"crop": result}