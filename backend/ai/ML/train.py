"""
train.py - Unified ML Training Pipeline
========================================
Run this script from the `backend/` directory:

    python -m ai.ML.train

Pipeline steps:
  1. Encode text columns → saves encoders.pkl
  2. Extract features & normalize → saves scaler.pkl
  3. Train Isolation Forest → saves isolation_forest.pkl    (unsupervised anomaly labels)
  4. Train RF Regressor    → saves random_forest_regressor.pkl  (normal spend baseline)
  5. Train RF Classifier   → saves random_forest_classifier.pkl (supervised fraud classifier)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score

from ai.ML.data.dataprocessing.encoding import encode_columns, TEXT_COLUMNS
from ai.ML.data.dataprocessing.preprocessing import extract_features, normalize

# ---------------------------------------------------------------------------
# Paths (all relative to backend/)
# ---------------------------------------------------------------------------
HERE        = os.path.dirname(__file__)
DATA_DIR    = os.path.join(HERE, 'data', 'data_2')
MODELS_DIR  = os.path.join(HERE, 'models')

RAW_FILE        = os.path.join(DATA_DIR, 'data.json')
ENCODED_FILE    = os.path.join(DATA_DIR, 'data_encoded.json')
PROCESSED_FILE  = os.path.join(DATA_DIR, 'data_processed.json')
LABELED_FILE    = os.path.join(DATA_DIR, 'data_labeled.json')

ENCODERS_PATH   = os.path.join(MODELS_DIR, 'encoders.pkl')
SCALER_PATH     = os.path.join(MODELS_DIR, 'scaler.pkl')
ISO_PATH        = os.path.join(MODELS_DIR, 'isolation_forest.pkl')
RFR_PATH        = os.path.join(MODELS_DIR, 'random_forest_regressor.pkl')
RFC_PATH        = os.path.join(MODELS_DIR, 'random_forest_classifier.pkl')

os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature lists — must match inference in prediction_service.py
# ---------------------------------------------------------------------------
IF_FEATURES = [
    'Transaction amount', 'Account balance', 'Salary (per month)',
    'Hour', 'DayOfWeek', 'Age', 'Is_Weekend', 'Is_Night',
    'Balance_to_Salary_Ratio', 'Transaction_to_Balance_Ratio',
    'Transaction Detail', 'Geological', 'Device Use',
    'Location', 'Working Status', 'Gender', 'Transaction Count'
]

RFR_FEATURES = [
    'Salary (per month)', 'Account balance', 'Transaction Count',
    'Working Status', 'Hour', 'DayOfWeek', 'Transaction Detail',
    'Location', 'Geological', 'Gender', 'Age', 'Is_Weekend', 'Is_Night',
    'Balance_to_Salary_Ratio', 'Transaction_to_Balance_Ratio'
]

RFC_FEATURES = IF_FEATURES  # same feature set as Isolation Forest


# ---------------------------------------------------------------------------
# Step 1: Encode
# ---------------------------------------------------------------------------
def step_encode(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[1/5] Encoding text columns...")
    df, encoders = encode_columns(df, TEXT_COLUMNS)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"    Encoders saved → {ENCODERS_PATH}")
    df.to_json(ENCODED_FILE, orient='records', indent=4, force_ascii=False)
    print(f"    Encoded data   → {ENCODED_FILE}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Feature engineering + normalization
# ---------------------------------------------------------------------------
def step_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/5] Extracting features and normalising...")
    df = extract_features(df)
    df = normalize(df, save_path=SCALER_PATH)
    df = df.fillna(0)
    df.to_json(PROCESSED_FILE, orient='records', indent=4, force_ascii=False)
    print(f"    Processed data → {PROCESSED_FILE}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Isolation Forest (unsupervised labels)
# ---------------------------------------------------------------------------
def step_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/5] Training Isolation Forest...")
    available = [c for c in IF_FEATURES if c in df.columns]
    X = df[available].fillna(0)

    model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
    model.fit(X)
    joblib.dump(model, ISO_PATH)
    print(f"    Model saved    → {ISO_PATH}")

    df['is_fraud']      = (model.predict(X) == -1).astype(int)
    df['anomaly_score'] = model.score_samples(X)
    fraud_count = df['is_fraud'].sum()
    print(f"    Labelled {fraud_count:,} / {len(df):,} transactions as fraud ({fraud_count/len(df)*100:.1f}%)")

    # Persist labeled data for supervised training
    df_out = df.copy()
    for col in ('Timestamp', 'DateTime'):
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str)
    df_out.to_json(LABELED_FILE, orient='records', indent=4, force_ascii=False)
    print(f"    Labeled data   → {LABELED_FILE}")
    return df


# ---------------------------------------------------------------------------
# Step 4: RF Regressor — learn normal spend habits
# ---------------------------------------------------------------------------
def step_rf_regressor(df: pd.DataFrame):
    print("\n[4/5] Training Random Forest Regressor (spend-habit baseline)...")
    available = [c for c in RFR_FEATURES if c in df.columns]
    X = df[available].fillna(0)
    y = df['Transaction amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, RFR_PATH)
    print(f"    Model saved    → {RFR_PATH}")

    preds = model.predict(X_test)
    print(f"    MAE : {mean_absolute_error(y_test, preds):.2f}")
    print(f"    R²  : {r2_score(y_test, preds):.4f}")


# ---------------------------------------------------------------------------
# Step 5: RF Classifier — supervised fraud classifier
# ---------------------------------------------------------------------------
def step_rf_classifier(df: pd.DataFrame):
    print("\n[5/5] Training Random Forest Classifier (fraud detector)...")
    available = [c for c in RFC_FEATURES if c in df.columns]
    X = df[available].fillna(0)
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, RFC_PATH)
    print(f"    Model saved    → {RFC_PATH}")

    preds = model.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, preds):.2%}")
    print(classification_report(y_test, preds, target_names=['Normal', 'Fraud']))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("  ML Training Pipeline")
    print("=" * 60)

    print(f"\nLoading raw data from: {RAW_FILE}")
    df = pd.read_json(RAW_FILE)
    print(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")

    df = step_encode(df)
    df = step_preprocess(df)
    df = step_isolation_forest(df)
    step_rf_regressor(df)
    step_rf_classifier(df)

    print("\n" + "=" * 60)
    print("  Training complete. All models saved to models/")
    print("=" * 60)
