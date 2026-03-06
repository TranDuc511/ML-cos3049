import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt


def load_data(file_path):
    return pd.read_json(file_path)


def prepare_features(df):
    feature_cols = [
        'Salary (per month)', 'Account balance', 'Transaction Count',
        'Working Status', 'Hour', 'DayOfWeek', 'Transaction Detail',
        'Location', 'Geological', 'Gender', 'Age', 'Is_Weekend', 'Is_Night',
        'Balance_to_Salary_Ratio', 'Tx_to_Balance_Ratio'
    ]

    available = [col for col in feature_cols if col in df.columns]
    X = df[available].copy()
    y = df['Transaction amount']

    X = X.fillna(X.median(numeric_only=True))
    return X, y, available


def train_and_evaluate(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"MSE: {mean_squared_error(y_test, preds):.2f}")
    print(f"R2:  {r2_score(y_test, preds):.4f}")

    importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances:")
    for name, score in importances:
        print(f"  {name:<25} {score:.4f}")

    return model, preds, y_test


def visualize(model, feature_names, y_test, preds):
    errors = y_test.values - preds

    # 1. Actual vs Predicted
    plt.figure()
    plt.scatter(y_test, preds, alpha=0.4, color='steelblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Transaction Amount')
    plt.ylabel('Predicted Transaction Amount')
    plt.title('Actual vs Predicted Transaction Amount')

    # 2. Feature Importances
    plt.figure()
    importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1])
    names, scores = zip(*importances)
    plt.barh(names, scores, color='teal')
    plt.xlabel('Importance Score')
    plt.title('Feature Importances (Customer Habits)')

    # 3. Prediction Error Distribution
    plt.figure()
    plt.hist(errors, bins=50, color='coral')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.title('Prediction Error Distribution')

    # 4. Residuals vs Predicted
    plt.figure()
    plt.scatter(preds, errors, alpha=0.4, color='mediumpurple')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Amount')
    plt.ylabel('Residual (Error)')
    plt.title('Residuals vs Predicted Amount')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data('ML/data/data_labeled.json')
    X, y, feature_names = prepare_features(df)
    model, preds, y_test = train_and_evaluate(X, y, feature_names)
    visualize(model, feature_names, y_test, preds)
