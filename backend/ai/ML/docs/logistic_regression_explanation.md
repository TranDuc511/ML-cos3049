# Logistic Regression: Code Explanation & Model Comparison

## How It Works

Logistic Regression predicts the **probability** that a transaction is fraud. It learns a linear boundary between classes using the sigmoid function:

$$P(\text{fraud}) = \frac{1}{1 + e^{-(w_1x_1 + w_2x_2 + \cdots + b)}}$$

Each feature gets a **coefficient (weight)** — positive weights push toward fraud, negative weights push toward normal.

---

## Code Walkthrough

### `load_data(file_path)`
```python
df = pd.read_json(file_path)
```
Reads the labeled dataset (`data_labeled.json`) into a DataFrame.

---

### `prepare_data(df)`
```python
feature_columns = ['Transaction amount', 'Account balance', ..., 'Transaction Count']
available = [col for col in feature_columns if col in df.columns]
X = df[available].copy()
y = df['is_fraud']
```
- Selects 17 features (numeric + encoded categoricals).
- `X` = features, `y` = fraud label (0 or 1).
- Filters to only columns that exist, preventing key errors.

---

### `train_model(X, y, feature_names)`
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
```
- **80/20 train-test split** with a fixed seed for reproducibility.
- `max_iter=1000` — allows the solver enough iterations to converge.
- Saves the trained model to `models/logistic_regression.pkl` via `joblib`.

```python
coef_df = pd.DataFrame({'Features': feature_names, 'Coefficients': model.coef_[0]})
            .sort_values('Coefficients', ascending=False, key=abs)
```
- Extracts feature **coefficients** sorted by absolute magnitude — the logistic regression equivalent of feature importance.

---

### `visualize(coef_df, X_test, y_test, predictions, model)`

| Plot | Purpose |
|------|---------|
| **Feature Coefficients (bar chart)** | Shows which features most influence fraud prediction |
| **Confusion Matrix** | True/false positives and negatives |
| **ROC Curve** | Trade-off between true positive rate and false positive rate |
| **Precision-Recall Curve** | Performance at different fraud detection thresholds |

Output saved to `visualization/logistic_output.png`.

---

### `__main__` entry point
```python
df = load_data(...)
X, y, columns = prepare_data(df)
coef_df, X_test, y_test, predictions, model = train_model(X, y, columns)
visualize(coef_df, X_test, y_test, predictions, model)
```
Runs the full model training process when executed as a script.

---

## Model Comparison: Logistic Regression vs. Random Forest Regressor

First, it's important to clarify the task type:
- **Logistic Regression** is natively built for **classification** tasks (predicting categories, like Fraud vs Normal) despite its name containing "Regression".
- **Random Forest Regressor** is natively built for **continuous regression** tasks (predicting numbers, like stock prices or anomaly scores).
- A fairer comparison for this project would be Logistic Regression vs Random Forest *Classifier*, but we will contrast the two approaches used.

| Aspect | Logistic Regression (Classifier) | Random Forest Regressor |
|--------|-------------------|---------------|
| **Task Purpose** | Predicts binary label: 0 (Normal) or 1 (Fraud) | Predicts a continuous value (e.g., amount, or anomaly score) |
| **Output Type** | Probabilities bounded between 0 and 1 | Unbounded numeric values |
| **Core Method** | Linear mathematical equation (hyperplane) | Ensemble average of 100 decision trees |
| **Interpretability** | ✅ High — coefficients show direct impact | ⚠️ Low — "black box" average of trees |
| **Feature insights** | Coefficients (sign indicates positive/negative effect) | Feature importances (magnitude of splits, no direction) |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC | Mean Squared Error (MSE), R-squared |

### Key Differences in Usage
- **Logistic Regression** is used when you need to output a strict category. It outputs `1` or `0`.
- **Random Forest Regressor** would be used if you wanted to predict continuous values, e.g., "What will be the exact transaction amount?" or a continuous anomaly score where a separate threshold must be defined manually.

For the pure task of Fraud Detection (a binary yes/no problem), **Logistic Regression** (or a Random Forest Classifier) is the technically correct tool, whereas a Regressor requires artificially mapping numeric outputs back to binary decisions.
