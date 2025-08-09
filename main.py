from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load("slr_model.joblib")

# Define FastAPI app
app = FastAPI()

# Define the input schema
class InputData(BaseModel):
    ENGINESIZE: float

@app.get("/")
def read_root():
    return {"message": "Hello, Guys! Welcome,To TJK'S ML World"}

@app.post("/predict")
def predict(data: InputData):
    # Convert the input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(input_df)

    # Convert NumPy type to regular float
    result = float(prediction[0])


    # Return a valid JSON response
    return {"prediction": result}

