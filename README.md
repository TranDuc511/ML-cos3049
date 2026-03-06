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
│   │   ├── data.json                               # Raw merged data
│   │   ├── data_encoded.json                       # Text fields encoded
│   │   ├── data_processed.json                     # Normalized & features added
│   │   └── data_labeled.json                       # Labeled by Isolation Forest
│   ├── dataprocessing/
│   │   ├── merge.py                                # Step 1: Merge raw data
│   │   ├── encoding.py                             # Step 2: Encode categories
│   │   └── preprocessing.py                        # Step 3: Extract features & normalize
│   ├── docs/
│   │   └── (Markdown explanations of models)
│   ├── src/
│   │   ├── isolationforest.py                      # Step 4: Anomaly detection
│   │   ├── random_forest.py                        # Step 5: Classification
│   │   └── rdregressor.py                          # Step 6: Regression
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

**Step 1: Merge Data**

```bash
python ML/dataprocessing/merge.py
```

**Step 2: Encode Categorical Features**

```bash
python ML/dataprocessing/encoding.py
```

**Step 3: Normalize & Extract Features**

```bash
python ML/dataprocessing/preprocessing.py
```

**Step 4: Generate Labels (Isolation Forest)**

```bash
python ML/src/isolationforest.py
```

**Step 5: Train Classifier (Random Forest)**

```bash
python ML/src/random_forest.py
```

**Step 6: Train Regressor (Random Forest Regressor)**

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
- [x] Regression (RandomForest Regressor) (Predict transaction amount bases on customer habit - )
- [x] Model Evaluation
- [ ] Visualization
- [x] Dataset processing code
- [ ] Notebook for each model

---

*3 Forest modes for real*
