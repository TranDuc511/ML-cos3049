import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
import matplotlib.pyplot as plt


def load_data(file_path):
    return pd.read_json(file_path)


def prepare_features(df):
    feature_cols = [
        'Salary (per month)', 'Account balance', 'Transaction Count',
        'Working Status', 'Hour', 'DayOfWeek', 'Transaction Detail',
        'Location', 'Geological', 'Gender', 'Age', 'Is_Weekend', 'Is_Night',
        'Balance_to_Salary_Ratio',  
        #  REMOVED: 'Transaction_to_Balance_Ratio' 
    ]

    available = [col for col in feature_cols if col in df.columns]
    X = df[available].copy()
    y = df['Transaction amount']

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    
    return X, y, available


def train_and_evaluate(X, y, feature_names):
    """Train Random Forest Regressor and evaluate performance."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_regressor.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"\n{'='*60}")
    print(f"Random Forest Regressor Performance")
    print(f"{'='*60}")
    print(f"MAE (Mean Absolute Error):  {mae:>15,.2f}")
    print(f"MSE (Mean Squared Error):   {mse:>15,.2f}")
    print(f"RMSE:                       {np.sqrt(mse):>15,.2f}")
    print(f"R² Score:                   {r2:>15.4f}")
    print(f"{'='*60}")

    # Feature importances (should NOT show Transaction_to_Balance_Ratio dominance)
    importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Feature Importances (legitimate patterns, no leakage):")
    print(f"{'Feature':<30} {'Importance':<15}")
    print("-" * 45)
    for name, score in importances[:10]:
        print(f"{name:<30} {score:>14.4f}")

    return model, preds, y_test, importances


def visualize(model, feature_names, y_test, preds, importances):
    """Visualize model performance and feature importance."""
    errors = y_test.values - preds

    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, preds, alpha=0.4, color='steelblue', s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Transaction Amount (VND)', fontsize=11)
    plt.ylabel('Predicted Transaction Amount (VND)', fontsize=11)
    plt.title('Actual vs Predicted Transaction Amount\n(Without Data Leakage)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. Feature Importances (Top 15)
    plt.figure(figsize=(10, 8))
    top_features = importances[:15]
    names, scores = zip(*top_features)
    plt.barh(names, scores, color='teal', alpha=0.8)
    plt.xlabel('Importance Score', fontsize=11)
    plt.title('Top 15 Feature Importances\n(Legitimate patterns, no leakage)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    # 3. Prediction Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', lw=2, label='Zero Error')
    plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')

    # 4. Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(preds, errors, alpha=0.4, color='mediumpurple', s=30)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicted Amount (VND)', fontsize=11)
    plt.ylabel('Residual / Error', fontsize=11)
    plt.title('Residuals vs Predicted Amount\n(Check for heteroscedasticity)', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    HERE = os.path.dirname(__file__)
    df = load_data(os.path.join(HERE, '..', 'data', 'data_2', 'data_labeled.json'))
    X, y, feature_names = prepare_features(df)
    model, preds, y_test = train_and_evaluate(X, y, feature_names)
    visualize(model, feature_names, y_test, preds)