import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

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
        'Balance_to_Salary_Ratio', 'Tx_to_Balance_Ratio',
        'Transaction Detail', 'Geological', 'Device Use', 
        'Location', 'Working Status', 'Gender', 'Transaction Count'
    ]

    available = [col for col in feature_columns if col in df.columns]
    X = df[available].copy()
    y = df['is_fraud']

    return X, y, available


def train_model(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Fraud']))

    importance_df = pd.DataFrame({
        'Features': feature_names,
        'Importances': model.feature_importances_
    }).sort_values('Importances', ascending=False)
    print(importance_df.head(5).to_string(index=False))

    return importance_df, X_test, y_test, predictions, model


def visualize(importance_df, X_test, y_test, predictions, model):
    importance_df.sort_values('Importances').plot(
        kind='barh', x='Features', y='Importances', legend=False, color='teal'
    )
    plt.title('Feature Importances (Random Forest)')
    plt.xlabel('Importance Score')

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=['Normal', 'Fraud'], cmap='Blues')

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('ROC Curve - Random Forest')

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title('Precision-Recall Curve')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data('ML/data/data_labeled.json')
    X, y, columns = prepare_data(df)
    importance_df, X_test, y_test, predictions, model = train_model(X, y, columns)
    visualize(importance_df, X_test, y_test, predictions, model)