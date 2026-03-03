DO NOT RUN ```run_pipeline.py```, require fixing

# Transaction Anomaly Detection System

This project implements a machine learning pipeline to detect and analyse anomalous financial transactions using **Isolation Forest** (unsupervised anomaly detection), **Random Forest Classifier** (supervised classification), and **Random Forest Regressor** (predicting transaction amounts).

## Features

- **Automated Anomaly Detection**: Uses Isolation Forest to identify suspicious transactions without prior labeling.
- **Supervised Classification**: Uses Random Forest to learn and classify the patterns of detected anomalies.
- **Regression Analysis**: Predicts the expected transaction amount using a Random Forest Regressor.
- **Explainable AI**: Provides feature importance to explain *why* a transaction is considered anomalous.

## Project Structure

```text
ML-cos3049/
├── ML/
│   ├── data/
│   │   ├── data.json                               # Raw transaction data
│   │   └── data_labeled.json                       # Data labeled by Isolation Forest
│   ├── docs/
│   │   ├── isolation_forest_explanation.md         # Explanation of Isolation Forest
│   │   ├── random_forest_explanation.md            # Explanation of Random Forest
│   │   └── random_forest_regressor_explanation.md  # Explanation of RF Regressor
│   ├── notebook/
│   │   ├── iso.ipynb                               # Isolation Forest notebook
│   │   ├── rdclassification.ipynb                  # Classification notebook
│   │   └── redregressor.ipynb                      # Regression notebook
│   ├── results/                                    # Saved model outputs
│   ├── src/
│   │   ├── isolation_forest_anomaly_detection.py   # Step 1: Anomaly detection & labeling
│   │   ├── random_forest.py                        # Step 2: Classification model
│   │   └── rdregressor.py                          # Step 3: Regression model
│   ├── visualizations/                             # Saved plots and charts
│   └── requirements.txt
├── run_pipeline.py
└── README.md
```

## Instructions

### 1. Understanding the Algorithms

Before running the code, read the simplified explanations in `ML/docs/`:

- [📖 Isolation Forest Explanation](ML/docs/isolation_forest_explanation.md)
- [📖 Random Forest Explanation](ML/docs/random_forest_explanation.md)
- [📖 Random Forest Regressor Explanation](ML/docs/random_forest_regressor_explanation.md)

### 2. Running the Code

#### Option A: Run the Entire Pipeline (Recommended)

Runs all steps (Detection → Classification → Regression) in one command:

```bash
python run_pipeline.py
```

#### Option B: Run Step-by-Step

**Step 1: Generate Labels (Isolation Forest)**

```bash
python ML/src/isolation_forest_anomaly_detection.py
```

**Step 2: Train Classifier (Random Forest)**

```bash
python ML/src/random_forest.py
```

**Step 3: Train Regressor (Random Forest Regressor)**

```bash
python ML/src/rdregressor.py
```

### 3. Expected Output

- Labeled dataset saved to `ML/data/data_labeled.json`
- Classification report with accuracy, precision, recall
- Regression metrics: MAE, MSE, R²
- Top 5 most important features for each model

## Requirements

```bash
pip install -r ML/requirements.txt
```

- Python 3.x
- pandas
- numpy
- scikit-learn

## Setup (Anaconda)

```bash
conda create -n ml python=3.10
conda activate ml
pip install pandas numpy scikit-learn
```

## Checklist

- [x] Clustering (IsolationForest) → Labeling
- [x] Classification (RandomForest)
- [] Regression (RandomForest Regressor) (need to review again)
- [x] Model Evaluation
- [ ] Visualization
- [ ] Dataset processing code
- [ ] Notebook for each model

---

*3 Forest modes for real*
