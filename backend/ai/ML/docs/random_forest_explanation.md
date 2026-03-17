# Random Forest: An Academic Overview

## Introduction

Random Forest is a machine learning algorithm used for both **classification** (categorizing data) and **regression** (predicting numbers). It belongs to the family of **ensemble learning** methods, which means it combines multiple smaller models to create a single, more powerful model.

The core idea is simple: instead of relying on one "decision tree," Random Forest builds a "forest" of many trees. The final result is determined by the majority vote of all trees.

## Core Concepts

### 1. Decision Trees

A decision tree is like a flowchart. It asks a series of Yes/No questions to reach a conclusion.

- **Example**: To decide if a transaction is fraudulent, a tree might ask: "Is the amount > $10,000?" -> If Yes, "Is the location new?" -> If Yes, "Fraud".

While intuitive, a single decision tree is prone to **overfitting**. It memorizes the training data too well and fails to generalize to new data.

### 2. Bagging (Bootstrap Aggregating)

To prevent overfitting, Random Forest uses a technique called **Bagging**.

- **Bootstrap**: The algorithm creates multiple subsets of the original data by sampling randomly with replacement.
- **Aggregating**: Each tree is trained on a different subset. The final prediction is an average (or vote) of all trees.

### 3. Random Feature Selection

In addition to randomizing data, Random Forest also randomizes **features**. at each split in a tree, the algorithm considers only a random subset of features (e.g., only "Amount" and "Time", ignoring "Location").

- This ensures that the trees are diverse and not correlated with each other.

## How It Works in Practice

1. **Training Phase**:
    - The algorithm builds $N$ decision trees (e.g., 100 trees).
    - Each tree learns from a random slice of the data.
2. **Prediction Phase**:
    - When a new data point arrives, it is passed through every tree.
    - Each tree gives a vote (e.g., Tree 1 says "Normal", Tree 2 says "Fraud").
    - The final output is the option with the most votes.

## Advantages and Disadvantages

| Advantages | Disadvantages |
| :--- | :--- |
| **High Accuracy**: robust against noise and outliers. | **Slow Prediction**: processing 100+ trees takes time. |
| **No Overfitting**: randomization keeps the model general. | **Black Box**: harder to interpret than a single tree. |
| **Feature Importance**: reveals which factors are most critical. | **Memory Intensive**: requires storing the entire forest. |

## Conclusion

Random Forest is a versatile and robust algorithm. By aggregating the wisdom of many "weak" learners (individual trees), it creates a "strong" learner capable of solving complex problems with high accuracy.
