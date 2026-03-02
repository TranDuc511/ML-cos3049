# Transaction Anomaly Detection System

This project implements a machine learning system to detect anomalous financial transactions using a combination of **Isolation Forest** (for unsupervised anomaly detection) and **Random Forest** (for supervised classification/explanation).

## Features

- **Automated Anomaly Detection**: Uses Isolation Forest to identify suspicious transactions without prior labeling.
- **Supervised Learning Integration**: Uses Random Forest to learn the patterns of detected anomalies.
- **Explainable AI**: Provides feature importance to explain *why* a transaction is considered anomalous.

## Project Structure

Algorithm_Inno/ML/
├── data/
│   ├── data.json           # Raw transaction data
│   └── data_labeled.json   # Data labeled by Isolation Forest
├── notebook/
│   ├── rdclassification.ipynb
│   |── rdregressor.ipynb
|   |── iso.ipynb
├── docs/
│   ├── random_forest_explanation.md    # Simlified explanation of Random Forest
│   └── isolation_forest_explanation.md # Simplified explanation of Isolation Forest
├── src/
│   ├── isolation_forest_anomaly_detection.py
│   └── random_forest.py
└── README.md

## Instructions

This project demonstrates two powerful machine learning algorithms: **Isolation Forest** (for finding anomalies) and **Random Forest** (for classifying them).

### 1. Understanding the Algorithms

Before running the code, please read the simplified explanations:

- [📖 Random Forest Explanation](docs/random_forest_explanation.md)
- [📖 Isolation Forest Explanation](docs/isolation_forest_explanation.md)

### 2. Running the Code

You have two ways to run this project:

#### Option A: Run the Entire Pipeline (Recommended)

You can run the entire process (Anomaly Detection -> Classification) with a single command from the `Algorithm_Inno` folder:

```bash
python run_pipeline.py
```

### 3. Expected Output

## Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy

## Create virtual enviroment

- install Anaconda
- open Anaconda Prompt
- conda create -n ml python=3.10
- conda activate ml
- pip install pandas numpy scikit-learn

## Checklist

[]Clustering (IsolationForest) -> [x] Labeling

[x]Classification (RandomForest)

[]Regression (RandomForest)

[x]Model Evaluation

[]Visualization

[]Dataset processing code

[]Notebook for each model

3 Forest modes for real
