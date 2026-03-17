import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

import joblib
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def load_data(file_path):
    df = pd.read_json(file_path)
    print(f"Loaded {len(df):,} rows.")
    return df


def prepare_data(df):
    feature_columns = [
        'Transaction amount', 'Account balance', 'Salary (per month)',
        'Hour', 'DayOfWeek', 'Age', 'Is_Weekend', 'Is_Night',
        'Balance_to_Salary_Ratio', 'Transaction_to_Balance_Ratio',
        'Transaction Detail', 'Geological', 'Device Use',
        'Location', 'Working Status', 'Gender', 'Transaction Count'
    ]

    available = [col for col in feature_columns if col in df.columns]
    X = df[available].copy()
    y = df['is_fraud']

    return X, y, available


def train_model(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'logistic_regression.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Fraud']))

    # Feature coefficients (logistic regression equivalent of feature importances)
    coef_df = pd.DataFrame({
        'Features': feature_names,
        'Coefficients': model.coef_[0]
    }).sort_values('Coefficients', ascending=False, key=abs)
    print(coef_df.head(5).to_string(index=False))

    return coef_df, X_test, y_test, predictions, model


def visualize(coef_df, X_test, y_test, predictions, model):
    # 1. Feature Coefficients
    coef_df.sort_values('Coefficients').plot(
        kind='barh', x='Features', y='Coefficients', legend=False, color='teal'
    )
    plt.title('Feature Coefficients (Logistic Regression)')
    plt.xlabel('Coefficient Value')

    # 2. Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=['Normal', 'Fraud'], cmap='Blues')

    # 3. ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('ROC Curve - Logistic Regression')

    # 4. Precision-Recall Curve
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title('Precision-Recall Curve')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'visualization', 'logistic_output.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    HERE = os.path.dirname(__file__)
    df = load_data(os.path.join(HERE, '..', 'data', 'data_2', 'data_labeled.json'))
    X, y, columns = prepare_data(df)
    coef_df, X_test, y_test, predictions, model = train_model(X, y, columns)
    visualize(coef_df, X_test, y_test, predictions, model)
