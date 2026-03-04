import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_data(file_path):
    
    print(f"Step 1: Reading {file_path}")
    
    data_frame = pd.read_json(file_path)
    print(f"Upload {len(data_frame)}")
    return data_frame


def prepared_date_rg(data_frame):
    print("Preparing data for regression")

    feature_columns = ['Account balance', 'Salary (per month)', 'Hour', 'DayOfWeek', 
        'Age', 'Transaction Detail', 'Geological', 
        'Location', 'Working Status', 'is_fraud' # We can use is_fraud as a feature now
    ]

    available_columns = [col for col in feature_columns if col in data_frame.columns]
        
    features = data_frame[available_columns].copy()
    target = data_frame['Transaction amount']

    
    text_columns = features.select_dtypes(include=['object']).columns
    if len(text_columns) > 0:
        for column in text_columns:
            encoder = LabelEncoder()
            features[column] = encoder.fit_transform(features[column].astype(str))
        
    return features, target, available_columns


def train_regression_model(features, target, feature_names):
    print("Step 3 Training Regression Model")
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # 3. USE RANDOM FOREST REGRESSOR
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Step 4 Evaluating Regression Model")
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)


    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")


    

if __name__ == "__main__":
    FILE_PATH = 'ML/data/data_labeled.json'
    
    df = load_data(FILE_PATH)
    if df is not None:
        X, y, columns = prepared_date_rg(df)
        train_regression_model(X, y, columns)
        print("Done")

