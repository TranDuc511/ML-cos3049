# Random Forest Regressor: An Academic Overview

## Introduction

Random Forest Regressor is a machine learning algorithm used to **predict continuous numerical values** (e.g., a transaction amount, a house price, or a salary). It is the regression counterpart to the Random Forest Classifier, and belongs to the same family of **ensemble learning** methods.

Instead of voting on a category (like "Fraud" or "Normal"), the trees in a regressor each predict a number, and the final output is the **average** of all those predictions.

## Core Concepts

### 1. Regression vs. Classification

The key distinction between the two types of tasks:

| | Classification | Regression |
| :--- | :--- | :--- |
| **Output** | A category (e.g., Fraud, Normal) | A number (e.g., \$450.75) |
| **Example** | "Is this transaction fraudulent?" | "How much will this transaction be?" |
| **Metric** | Accuracy, F1-Score | MAE, MSE, R² |

### 2. How Prediction is Aggregated

In regression, each decision tree in the forest independently predicts a numeric value for a data point. The final prediction is calculated as:

$$\hat{y} = \frac{1}{N} \sum_{i=1}^{N} T_i(x)$$

Where $N$ is the number of trees and $T_i(x)$ is the prediction of the $i$-th tree for input $x$.

### 3. Key Evaluation Metrics

Since there is no single "correct" answer in regression (only closer or further estimates), different metrics are used to measure how well the model performs:

- **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values. Easy to interpret — it is measured in the same unit as the target (e.g., dollars).
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- **Mean Squared Error (MSE)**: Like MAE, but squares the errors first. This penalizes large errors more heavily.
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **R-squared (R²)**: Measures how much of the variance in the target variable the model explains. A score of 1.0 is perfect; a score of 0.0 means the model is no better than predicting the mean.
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

## How It Works in Practice

1. **Training Phase**:
    - The algorithm builds $N$ decision trees (e.g., 100 trees).
    - Each tree is trained on a random subset of the data (Bagging) and uses a random subset of features at each split.
    - Each tree learns to map input features (e.g., `Account balance`, `Salary`, `Hour`) to the target value (`Transaction amount`).
2. **Prediction Phase**:
    - A new data point is passed through all 100 trees.
    - Each tree independently outputs a predicted number.
    - The final prediction is the **average** of all 100 numbers.

## Advantages and Disadvantages

| Advantages | Disadvantages |
| :--- | :--- |
| **No Feature Scaling Needed**: works directly with raw numerical values. | **Slow Prediction**: averaging 100+ trees takes time. |
| **Robust to Outliers**: averaging across many trees smooths out extreme predictions. | **Black Box**: hard to explain why a specific number was predicted. |
| **Handles Mixed Data**: works with both numerical and categorical features. | **Memory Intensive**: stores the full forest in memory. |
| **Feature Importance**: reveals which features most influence the predicted value. | **Extrapolation**: cannot predict values outside the range seen during training. |

## Application in This Project

In this project, the Random Forest Regressor is used to predict `Transaction amount` based on features such as:

- `Account balance`
- `Salary (per month)`
- `Hour`, `DayOfWeek`
- `Transaction Detail`, `Geological`, `Location`
- `Working Status`
- `is_fraud` (used as a feature — a fraudulent transaction may have a different value pattern)

The model is evaluated using **MAE**, **MSE**, and **R²** to measure how accurately it can estimate the transaction amount.

## Conclusion

The Random Forest Regressor extends the power and robustness of the ensemble approach to numerical prediction tasks. By averaging the outputs of many diverse trees, it produces stable and accurate estimates, making it one of the most practical and widely-used regression algorithms for tabular data.
