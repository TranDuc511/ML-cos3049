# Transaction Anomaly Detection API (FastAPI)

## 📌 Overview
This folder contains the FastAPI backend for the Transaction Anomaly Detection system. It acts as the bridge between your web frontend (React) and your trained machine learning models (stored in `backend/ai/ML/models`). It analyzes real-time financial transactions and classifies them as normal or fraudulent.

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have the required Python libraries installed (this backend uses the same environment as your ML training phase):
```bash
pip install fastapi uvicorn pydantic pandas scikit-learn joblib
```

### 2. Running the Server
1. Open a terminal directly in this `webapp` folder:
   ```bash
   cd backend/webapp
   ```
2. Start the live-reloading FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
3. The server will successfully boot up on `http://127.0.0.1:8000`.

---

## 📡 API Endpoints

### 🟢 1. Interactive Documentation (Swagger UI)
* **URL:** `http://127.0.0.1:8000/docs`
* **Description:** FastAPIs automatically generated testing platform. You can interactively test the endpoints here directly from your browser.

### 🧠 2. Fraud Prediction Engine
* **`POST /api/predicts`** 
  Takes a JSON payload containing `customer` and `transaction` characteristics. 
  Behind the scenes, the API handles:
  1. Merging objects into a Pandas DataFrame.
  2. Categorical label encoding (`encoders.pkl`).
  3. Feature extraction and min-max scaling (`scaler.pkl`).
  4. Execution against 3 loaded models: *Isolation Forest*, *Random Forest Regressor*, and *Random Forest Classifier*.
  
  **Returns:** An aggregated `is_fraud` boolean (using majority voting) and the exact decision breakdown from each model.

### 📊 3. History & Statistics (Placeholder)
*(Currently configured with empty placeholder routers; pending database integration)*
* `GET /api/history` - Shows recent transactions
* `DELETE /api/history` - Clears database history
* `GET /api/stats/summary` - Aggregates total transactions, fraud counts, and rates
* `GET /api/stats/fraud_by_hour` - Returns hourly bar chart dataset
* `GET /api/stats/history_trend` - Returns history line-chart dataset
* `GET /api/stats/amount_distribution` - Returns histogram bins configuration

---

## 🧪 Quick Test Sandbox
There is an integrated python sandbox you can use to simulate a frontend React request. While the server is running in one terminal, open another terminal and execute:
```bash
python test_api.py
```
This will instantly beam an example transaction into the anomaly detection pipeline and print out the backend's verdict!
