# Transaction Anomaly Detection System

This project implements a machine learning pipeline to detect and analyze anomalous financial transactions using **Isolation Forest** (unsupervised anomaly detection), **Random Forest Classifier** (supervised classification), and **Random Forest Regressor** (predicting transaction amounts).

## 1. Environment Setup

To ensure reproducibility, we use `conda` to manage the project's environment. Follow these step-by-step instructions to set up your environment:

1. **Create the Conda Environment:**
   Run the following command to create a new environment named `ml` with Python 3.10:
   ```bash
   conda create -n ml python=3.10 -y
   ```

2. **Activate the Environment:**
   ```bash
   conda activate ml
   ```

3. **Install Required Packages:**
   You can either install the packages manually using `pip`:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
   Or install directly from the project's requirements file:
   ```bash
   pip install -r ML/requirements.txt
   ```

## 2. Data Processing

The data processing pipeline prepares the raw dataset for model training. It is broken down into three sequential steps. Ensure you run these from the root directory:

1. **Merge Data**: Combines multiple raw data sources into a single working dataset (`ML/data/data.json`).
   ```bash
   python ML/dataprocessing/merge.py
   ```
2. **Encoding**: Converts categorical and text features (like Gender, Location, Working Status) into numerical representations (`ML/data/data_encoded.json`).
   ```bash
   python ML/dataprocessing/encoding.py
   ```
3. **Preprocessing & Feature Extraction**: Normalizes numerical continuous values (e.g., using `MinMaxScaler`) and extracts new predictive features, such as `Age` from Date of Birth, time-based flags (`Is_Weekend`, `Is_Night`), and financial ratios (`Balance_to_Salary_Ratio`, `Tx_to_Balance_Ratio`). The final dataset is saved to `ML/data/data_processed.json`.
   ```bash
   python ML/dataprocessing/preprocessing.py
   ```

## 3. Model Training

You can train the models using the preprocessed data. The training pipeline consists of three main components: Anomaly Detection, Classification, and Regression. 

*(Alternatively, to run the entire data processing and model training pipeline automatically, you can simply run: `python run_pipeline.py`)*

### Step 3.1: Anomaly Detection (Isolation Forest)
This step uses an unsupervised **Isolation Forest** to identify suspicious transactions without prior labeling. It assigns an `anomaly_score` and an `is_fraud` label (0 for Normal, 1 for Fraud) to each transaction.
* **Command:**
  ```bash
  python ML/src/isolationforest.py
  ```
* **Output:** Saves the labeled dataset to `ML/data/data_labeled.json` and displays anomaly distribution visualizations.

### Step 3.2: Fraud Classification (Random Forest Classifier)
Trains a supervised **Random Forest Classifier** using the dataset generated in Step 3.1 to learn the specific patterns of the detected anomalies.
* **Command:**
  ```bash
  python ML/src/random_forest.py
  ```
* **Outputs:** Prints Model Accuracy, Classification Report (Precision, Recall), Feature Importances, and displays the Confusion Matrix and ROC Curve.

### Step 3.3: Spending Prediction (Random Forest Regressor)
Trains a **Random Forest Regressor** to predict the expected transaction amount based on customer habits and behavior.
* **Command:**
  ```bash
  python ML/src/rdregressor.py
  ```
* **Outputs:** Prints evaluation metrics (MAE, MSE, R²) and visualizes Prediction Errors and Feature Importances.

## 4. Prediction

To use the trained models to make predictions on new, unseen data, you must first ensure the new data undergoes the exact same **Data Processing** steps (encoding, feature extraction, and scaling) as the training data.

Here is a clear python example illustrating how to make predictions using the trained classification model:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Load the pre-trained model (assuming the model is saved or currently in memory)
# Note: To persist the model across sessions in your own script, use the 'joblib' library:
# import joblib
# joblib.dump(model, 'rf_model.pkl') # Save during training
# model = joblib.load('rf_model.pkl') # Load for prediction

# 2. Prepare your new transaction data
new_data = pd.DataFrame([{
    'Transaction amount': 0.85, # Normalized
    'Account balance': 0.12,    # Normalized
    'Salary (per month)': 0.45, # Normalized
    'Hour': 14,
    'DayOfWeek': 2,
    'Age': 0.30,                # Normalized
    'Is_Weekend': 0,
    'Is_Night': 0,
    'Balance_to_Salary_Ratio': 0.26, # Normalized
    'Tx_to_Balance_Ratio': 7.08,     # Normalized
    'Transaction Detail': 3,    # Encoded
    'Geological': 1,            # Encoded
    'Device Use': 2,            # Encoded
    'Location': 4,              # Encoded
    'Working Status': 1,        # Encoded
    'Gender': 0,                # Encoded
    'Transaction Count': 5
}])

# 3. Ensure the columns match the exact features the model was trained on
expected_features = [
    'Transaction amount', 'Account balance', 'Salary (per month)',
    'Hour', 'DayOfWeek', 'Age', 'Is_Weekend', 'Is_Night',
    'Balance_to_Salary_Ratio', 'Tx_to_Balance_Ratio',
    'Transaction Detail', 'Geological', 'Device Use', 
    'Location', 'Working Status', 'Gender', 'Transaction Count'
]
X_new = new_data[expected_features]

# 4. Make a prediction
prediction = model.predict(X_new)

# 5. Interpret the result
if prediction[0] == 1:
    print("WARNING: This transaction is predicted as FRAUDULENT.")
else:
    print("This transaction is predicted as NORMAL.")
```

For the regression model (`rdregressor.py`), the usage is essentially identical. Simply replace the classification model with your trained regressor, and the `.predict()` function will output the **estimated transaction amount**.

---
*For more theoretical details on the chosen algorithms, please refer to the markdown files in the `ML/docs/` directory.*
