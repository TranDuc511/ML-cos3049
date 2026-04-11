# Backend - Transaction Anomaly Detection

This directory contains the machine learning pipeline (`ai/ML/`) and the FastAPI REST API server (`webapp/`).

## Prerequisites
- Python installed.

## 1. Installation

Open your terminal in the `backend/` directory and install the required packages:

```bash
pip install -r ai/ML/requirements.txt
pip install -r webapp/requirements.txt
```

## 2. Model Training (Optional)

If you need to re-train the models, ensure your data is placed at `ai/ML/data/data_2/data.json`. Then, from the `backend/` directory, run:

```bash
python -m ai.ML.train
```

## 3. Running the API Server

Change into the `webapp/` directory and start the server:

```bash
cd webapp
uvicorn main:app --reload
```

The server runs at `http://127.0.0.1:8000`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Detection Logic

The system scores each transaction using Isolation Forest, Random Forest Regressor, and Random Forest Classifier. A majority vote determines the final fraud verdict.