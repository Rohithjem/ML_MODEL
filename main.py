from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("iris_model.pkl")

# Define request schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Iris prediction API is running 🚀"}

@app.post("/predict")
def predict(input: IrisInput):
    # Convert input to DataFrame
    data = pd.DataFrame([input.dict()])
    
    # Make prediction
    pred_class = model.predict(data)[0]
    pred_proba = model.predict_proba(data).tolist()[0]

    return {
        "prediction_class": int(pred_class),
        "prediction_confidence": pred_proba[pred_class]
    }
