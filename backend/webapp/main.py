from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(title="ML Prediction Service")

# 1. Path to your .pkl file
MODEL_PATH = r"C:\Users\Admin\Documents\ML-cos3049\backend\ai\ML\models\random_forest_classifier.pkl"

# 2. Load the model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
else:
    model = None
    print(f"Error: Model file not found at {MODEL_PATH}")

# 3. Define the data structure (matching your training features)
class TransactionData(BaseModel):
    amount: float
    balance: float
    salary: float
    hour: int
    day_of_week: int
    age: int
    is_weekend: int
    is_night: int

@app.post("/predict")
async def predict(data: TransactionData):
    if model is None:
        return {"error": "Model not loaded"}

    # Create a DataFrame for prediction (must match training column order)
    input_df = pd.DataFrame([{
        'Transaction amount': data.amount,
        'Account balance': data.balance,
        'Salary (per month)': data.salary,
        'Hour': data.hour,
        'DayOfWeek': data.day_of_week,
        'Age': data.age,
        'Is_Weekend': data.is_weekend,
        'Is_Night': data.is_night
    }])

    # 4. Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    # 5. Return result
    return {
        "is_fraud": int(prediction[0]),
        "probability_fraud": float(probability[0][1]),
        "status": "Success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
