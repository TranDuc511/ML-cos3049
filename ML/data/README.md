# ML Data Pipeline Scripts

This folder contains scripts for generating synthetic data (`datacollecting/`) as well as cleaning, encoding, and enriching the raw data (`dataprocessing/`) before it is used by the ML models.

## 1. Data Collection (`datacollecting/`)

### 1.1 `customer.py`

Generates synthetic customer accounts with initial balances and demographic information.

- **Writes:** `ML/data/customers.json`

```bash
python ML/data/datacollecting/customer.py
```

---

### 1.2 `transaction.py`

Generates synthetic transactions between the generated customers and safely updates their transaction counts.

- **Reads:** `ML/data/customers.json`
- **Writes:** `ML/data/transaction.json`
- **Modifies:** `ML/data/customers.json` (updates transaction counts)

```bash
python ML/data/datacollecting/transaction.py
```

---

## 2. Data Processing (`dataprocessing/`)

### 2.1 `merge.py`

Joins the customer and transaction datasets into a single file.

- **Reads:** `ML/data/customers.json`, `ML/data/transaction.json`
- **Writes:** `ML/data/data.json`

```bash
python ML/data/dataprocessing/merge.py
```

---

### 2.2 `encoding.py`

Converts text (categorical) columns into numbers using Label Encoding.

- **Reads:** `ML/data/data.json`
- **Writes:** `ML/data/data_encoded.json`

```bash
python ML/data/dataprocessing/encoding.py
```

Columns encoded: `Transaction Detail`, `Geological`, `Device Use`, `Gender`, `Location`, `Working Status`

---

### 2.3 `preprocessing.py`

Normalizes numeric columns and extracts new features from existing ones.

- **Reads:** `ML/data/data_encoded.json`
- **Writes:** `ML/data/data_processed.json`

```bash
python ML/data/dataprocessing/preprocessing.py
```

New features: `Age`, `Is_Weekend`, `Is_Night`, `Balance_to_Salary_Ratio`, `Tx_to_Balance_Ratio`

---

## Run Order

These scripts must be run **in order** to generate and process the data before the ML pipeline:

```text
customer.py → transaction.py → merge.py → encoding.py → preprocessing.py
```

Or run everything (including data processing and model training) at once from the project root:

```bash
python run_pipeline.py
```
