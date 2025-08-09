from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("slr_model.joblib")

# Define FastAPI app
app = FastAPI()

# Define the input schema with custom name
class InputData(BaseModel):
    TJK_ENGINESIZE: float

@app.get("/")
def read_root():
    return {"message": "Hello, Guys! Welcome, To TJK'S ML World"}

@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Rename to match model's expected column
    input_df.rename(columns={"TJK_ENGINESIZE": "ENGINESIZE"}, inplace=True)

    # Make prediction
    prediction = model.predict(input_df)

    # Convert NumPy type to regular float
    result = float(prediction[0])

    # Return with your custom output key
    return {"TJKPREDICTION": result}
