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
├── docs/
│   ├── random_forest_explanation.md    # Simlified explanation of Random Forest
│   └── isolation_forest_explanation.md # Simplified explanation of Isolation Forest
├── src/
│   ├── isolation_forest_anomaly_detection.py
│   └── random_forest.py
└── README.md

```

## Instructions for Students

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

#### Option B: Run Step-by-Step
If you want to understand each step individually, run them in this order from the `Algorithm_Inno` folder:

**Step 1: Generate Labels (Isolation Forest)**
This script reads the raw data, identifies anomalies, and saves a labeled dataset.
```bash
python ML/src/isolation_forest_anomaly_detection.py
```

**Step 2: Train Classifier (Random Forest)**
This script reads the labeled dataset and trains a model to understand the anomalies.
```bash
python ML/src/random_forest.py
```

### 3. Expected Output

- **Isolation Forest**: Will print the top 5 most anomalous transactions.
- **Random Forest**: Will print the accuracy of the model and the top 5 features that contribute to the anomaly (e.g., Transaction Amount, Location).

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
[x]Clustering (IsolationForest) -> [x] Labeling
[x]Classification (RandomForest)
[]Regression
[]Model Evaluation