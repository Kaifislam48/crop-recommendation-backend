import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("Crop_recommendation.csv")

le = LabelEncoder()
le.fit(df["label"])

joblib.dump(le, "model/label_encoder.pkl")

print("Encoder created")