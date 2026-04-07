"""
prediction_service.py
======================
Loads the three pre-trained ML models and provides process_and_predict().

Expected runtime working directory: backend/
  → Run the FastAPI server as:  uvicorn main:app --reload
  → Then `ai` and its sub-packages are importable without sys.path tricks.
"""

import os
import joblib
import pandas as pd

from ai.ML.data.dataprocessing.preprocessing import extract_features

# ---------------------------------------------------------------------------
# Model paths (resolved relative to this file → backend/webapp/services/)
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(__file__)
MODELS_DIR  = os.path.abspath(os.path.join(_HERE, '..', '..', 'ai', 'ML', 'models'))

ENCODERS_PATH = os.path.join(MODELS_DIR, 'encoders.pkl')
SCALER_PATH   = os.path.join(MODELS_DIR, 'scaler.pkl')
ISO_PATH      = os.path.join(MODELS_DIR, 'isolation_forest.pkl')
RFR_PATH      = os.path.join(MODELS_DIR, 'random_forest_regressor.pkl')
RFC_PATH      = os.path.join(MODELS_DIR, 'random_forest_classifier.pkl')

# ---------------------------------------------------------------------------
# Load artefacts once at startup (graceful degradation if missing)
# ---------------------------------------------------------------------------
def _load(path, label):
    try:
        obj = joblib.load(path)
        print(f"[prediction_service] Loaded {label} from {path}")
        return obj
    except FileNotFoundError:
        print(f"[prediction_service] WARNING: {label} not found at {path}. Prediction will be skipped.")
        return None

encoders    = _load(ENCODERS_PATH, "encoders")
scaler      = _load(SCALER_PATH,   "scaler")
model_iso   = _load(ISO_PATH,      "isolation_forest")
model_rf_reg = _load(RFR_PATH,     "random_forest_regressor")
model_rf    = _load(RFC_PATH,      "random_forest_classifier")

# ---------------------------------------------------------------------------
# Text columns that must be label-encoded
# ---------------------------------------------------------------------------
TEXT_COLUMNS = [
    'Transaction Detail', 'Geological', 'Device Use',
    'Gender', 'Location', 'Working Status',
]

# Columns to drop before feeding features to any model
_COLS_TO_DROP = [
    'Customer ID', 'Receiver Account ID', 'Sender Account ID',
    'Timestamp', 'Transaction ID',
]

# Numeric columns that the scaler was fitted on (order must match training)
_COLS_TO_SCALE = [
    'Transaction amount', 'Account balance', 'Salary (per month)',
    'Age', 'Balance_to_Salary_Ratio', 'Transaction_to_Balance_Ratio',
]

# Index of 'Transaction amount' inside _COLS_TO_SCALE — used for inverse transform
_TX_AMOUNT_IDX = _COLS_TO_SCALE.index('Transaction amount')


def _inverse_amount(scaled_value: float) -> float:
    """Convert a scaled Transaction amount (0–1) back to original VND scale."""
    if scaler is None:
        return scaled_value
    col_min = scaler.data_min_[_TX_AMOUNT_IDX]
    col_max = scaler.data_max_[_TX_AMOUNT_IDX]
    return float(scaled_value * (col_max - col_min) + col_min)


# ---------------------------------------------------------------------------
# Helper: align a DataFrame to exactly the features a model expects
# ---------------------------------------------------------------------------
def _align(model, df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with exactly the columns the model was trained on."""
    if not hasattr(model, 'feature_names_in_'):
        return df
    expected = list(model.feature_names_in_)
    tmp = df.copy()
    for f in expected:
        if f not in tmp.columns:
            tmp[f] = 0
    return tmp[expected]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def process_and_predict(customer_data: dict, transaction_data: dict) -> dict:
    """
    Merge customer + transaction dicts, run the full preprocessing pipeline,
    and return a fraud verdict with individual model votes.

    Parameters
    ----------
    customer_data    : dict with keys matching training schema (e.g. "Customer ID", "Gender", …)
    transaction_data : dict with keys matching training schema (e.g. "Transaction amount", …)

    Returns
    -------
    {
        "is_fraud": bool,
        "votes": {
            "isolation_forest": 0 | 1,
            "random_forest_regressor": 0 | 1,
            "random_forest_classifier": 0 | 1,
        }
    }
    """
    # Default response — returned as-is if models are not loaded
    result = {
        "is_fraud": False,
        "predicted_amount": 0.0,
        "votes": {
            "isolation_forest": 0,
            "random_forest_regressor": 0,
            "random_forest_classifier": 0,
        },
    }

    if model_iso is None:
        return result   # Models not trained / loaded yet

    # ------------------------------------------------------------------
    # 1. Merge into a single-row DataFrame
    # ------------------------------------------------------------------
    combined = {**customer_data, **transaction_data}
    df = pd.DataFrame([combined])

    # ------------------------------------------------------------------
    # 2. Encode text columns with the fitted LabelEncoders
    # ------------------------------------------------------------------
    if encoders:
        for col in TEXT_COLUMNS:
            if col in df.columns and col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Unseen category → assign -1 (out-of-distribution signal)
                    df[col] = -1

    # ------------------------------------------------------------------
    # 3. Feature engineering (Age, Hour, ratios, etc.)
    # ------------------------------------------------------------------
    df = extract_features(df)

    # ------------------------------------------------------------------
    # 4. Drop raw ID / timestamp columns not used as model features
    # ------------------------------------------------------------------
    for col in _COLS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure Transaction Count exists (not present for single-tx inference)
    if 'Transaction Count' not in df.columns:
        df['Transaction Count'] = 1

    df = df.fillna(0)

    # ------------------------------------------------------------------
    # 5. Scale numeric columns with the fitted scaler
    # ------------------------------------------------------------------
    if scaler:
        available = [c for c in _COLS_TO_SCALE if c in df.columns]
        if available:
            df[available] = scaler.transform(df[available])

    # ------------------------------------------------------------------
    # 6. Run each model on its expected feature subset
    # ------------------------------------------------------------------
    # --- Isolation Forest ---
    p_iso = model_iso.predict(_align(model_iso, df))[0]
    is_fraud_iso = int(p_iso <= 0)   # -1 → anomaly

    # --- RF Regressor: predict normal spend, inverse-transform to VND, compare ---
    raw_amount        = float(combined.get('Transaction amount', 0))
    p_rf_reg_scaled   = model_rf_reg.predict(_align(model_rf_reg, df))[0]
    p_rf_reg          = _inverse_amount(p_rf_reg_scaled)   # now in real VND
    is_fraud_rfr      = int(raw_amount > p_rf_reg * 3)

    # --- RF Classifier ---
    p_rf         = model_rf.predict(_align(model_rf, df))[0]
    is_fraud_rfc = int(p_rf == 1)

    # ------------------------------------------------------------------
    # 7. Majority vote (≥ 2 of 3 models flag fraud → fraud)
    # ------------------------------------------------------------------
    fraud_score = is_fraud_iso + is_fraud_rfr + is_fraud_rfc
    result["is_fraud"]  = bool(fraud_score >= 2)
    result["predicted_amount"] = round(float(p_rf_reg), 2)
    result["votes"] = {
        "isolation_forest":        is_fraud_iso,
        "random_forest_regressor": is_fraud_rfr,
        "random_forest_classifier": is_fraud_rfc,
    }

    return result
