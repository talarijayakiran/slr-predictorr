import joblib
import pandas as pd

model = joblib.load("slr_model.joblib")

def predict(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0][0]

if __name__ == "__main__":
    sample = {"ENGINESIZE": 2.5}
    result = predict(sample)
    print("TJKPredicted CO2 Emission:", result)
