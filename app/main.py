from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Real Estate Valuation API")

# Load model globally so it's ready for requests
model = joblib.load("models/rf_model.pkl")

class PropertyData(BaseModel):
    bedrooms: int
    bathrooms: float
    square_feet: int
    year_built: int

@app.post("/predict")
def predict_price(prop: PropertyData):
    # Convert incoming data to a dataframe row
    input_data = pd.DataFrame([prop.dict()])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return {
        "estimated_price": round(prediction, 2),
        "currency": "USD"
    }