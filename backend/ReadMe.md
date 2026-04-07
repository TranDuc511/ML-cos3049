# Backend — Transaction Anomaly Detection

This directory contains two sub-projects:

| Directory | Purpose |
|-----------|---------|
| `ai/ML/`  | Machine learning pipeline — data processing, model training, and saved artefacts |
| `webapp/` | FastAPI REST API server that serves prediction and history endpoints |

---

## Project Structure

```
backend/
├── ai/
│   └── ML/
│       ├── data/
│       │   ├── data_2/          # Processed JSON datasets (gitignored)
│       │   └── dataprocessing/  # encoding.py, preprocessing.py
│       ├── models/              # Saved .pkl model files (gitignored)
│       ├── src/                 # Individual model training scripts
│       └── train.py             # ← Unified training pipeline (run this)
└── webapp/
    ├── main.py                  # FastAPI app entry point
    ├── requirements.txt         # Server dependencies
    ├── routes/                  # API route handlers
    └── services/                # Business logic (prediction, storage)
```

---

## Quick Start

### 1. Install dependencies

```bash
# AI / training dependencies
pip install -r ai/ML/requirements.txt

# Web server dependencies
pip install -r webapp/requirements.txt
```

### 2. Train the models

> **Prerequisite:** Place the raw merged dataset at `ai/ML/data/data_2/data.json`
> (see `ai/ML/data/README.md` for the expected schema).

Run the unified pipeline from the `backend/` directory:

```bash
python -m ai.ML.train
```

This will:
1. Encode text columns and save `models/encoders.pkl`
2. Extract features and scale numerics, saving `models/scaler.pkl`
3. Train an **Isolation Forest** (unsupervised anomaly labelling) → `models/isolation_forest.pkl`
4. Train a **Random Forest Regressor** (spend-habit baseline) → `models/random_forest_regressor.pkl`
5. Train a **Random Forest Classifier** (supervised fraud detector) → `models/random_forest_classifier.pkl`

### 3. Start the API server

```bash
cd webapp
uvicorn main:app --reload
```

The server starts at **http://127.0.0.1:8000**.

Interactive docs: **http://127.0.0.1:8000/docs**

---

## Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predicts` | Predict fraud for a single transaction |
| `GET`  | `/api/history`  | List last 100 transactions (current session) |
| `DELETE` | `/api/history` | Clear session history |
| `GET`  | `/api/stats/summary` | Totals and fraud rate |
| `GET`  | `/api/stats/fraud_by_hour` | Hourly fraud counts |
| `GET`  | `/api/stats/history_trend` | Daily transaction counts |
| `GET`  | `/api/stats/amount_distribution` | Transaction amount distribution |

See `webapp/api_endpoints.md` for full request/response schemas.

---

## Fraud Detection Logic

Each incoming transaction is scored by **three models in parallel**:

1. **Isolation Forest** — flags transactions that look like statistical outliers.
2. **RF Regressor** — predicts the expected spend amount for this customer profile; flags if actual spend > 3× predicted.
3. **RF Classifier** — supervised classifier trained on the anomaly labels produced by Isolation Forest.

A **majority vote** (≥ 2 of 3) determines the final `is_fraud` verdict.