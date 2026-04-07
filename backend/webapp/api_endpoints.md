# Transaction Anomaly Detection API Documentation

This document outlines the available REST API endpoints for the Transaction Anomaly Detection backend.

> [!IMPORTANT]
> **Data Storage:** This backend currently uses **in-memory storage**. All transaction history and statistics will be reset whenever the FastAPI server is restarted.

---

## 🚀 Swagger UI

FastAPI provides automatic interactive documentation. When the server is running, you can access the Swagger UI and test endpoints directly from the browser:

* **URL:** `http://127.0.0.1:8000/docs`
* **Alternate (ReDoc):** `http://127.0.0.1:8000/redoc`

---

## 🧠 Prediction Endpoints

### `POST /api/predicts`

Takes a JSON payload containing `customer` and `transaction` information. Merges this data, scales it, runs it against 3 pre-trained models, and determines if the transaction is fraudulent. Results are saved in memory for the current session.

**Request Body (JSON):**

```json
{
  "customer": {
    "Customer ID": "C6655",
    "Date of Birth": "1990-05-12",
    "Gender": "M",
    "Location": "Hanoi",
    "Working Status": "Employed",
    "Salary (per month)": 15000000
  },
  "transaction": {
    "Transaction ID": "TX10023",
    "Timestamp": "2026-03-23 14:00:00",
    "Sender Account ID": "C6655",
    "Receiver Account ID": "R999",
    "Transaction amount": 500000,
    "Transaction Detail": "Payment",
    "Geological": "10.762622, 106.660172",
    "Device Use": "Mobile",
    "Account balance": 25000000
  }
}
```

**Success Response (200 OK):**

```json
{
  "status": "success",
  "data": {
    "is_fraud": false,
    "votes": {
      "isolation_forest": 0,
      "random_forest_regressor": 0,
      "random_forest_classifier": 0
    }
  }
}
```

---

## 📊 History Endpoints

### `GET /api/history`

Retrieves a list of the 100 most recent transactions from the current session.

**Success Response:**

```json
{
  "status": "success",
  "data": [
    {
      "transaction_id": "TX10023",
      "timestamp": "2026-03-23 14:00:00",
      "amount": 500000.0,
      "device_use": "Mobile",
      "is_fraud": 0,
      "votes": { ... }
    }
  ]
}
```

### `DELETE /api/history`

Clears all transaction history stored in the current session's memory.

**Success Response:**

```json
{
  "status": "success",
  "data": "History cleared."
}
```

---

## 📈 Statistics & Analytics Endpoints

### `GET /api/stats/summary`

Returns aggregated totals and fraud rates for the current session.

**Success Response:**

```json
{
  "status": "success",
  "data": {
    "total_transactions": 1,
    "fraud_count": 0,
    "fraud_rate": 0.0
  }
}
```

### `GET /api/stats/fraud_by_hour`

Returns hourly fraud counts for the session.

### `GET /api/stats/history_trend`

Returns daily transaction counts for the session.

### `GET /api/stats/amount_distribution`

Returns distribution of transaction amounts across predefined bins.
