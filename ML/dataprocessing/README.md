# dataprocessing/

This folder contains scripts that clean, encode, and enrich the raw data before it is used by the ML models.

## Scripts

### 1. `merge.py`

Joins the customer and transaction datasets into a single file.

- **Reads:** `ML/datacollecting/customers.json`, `ML/datacollecting/transaction.json`
- **Writes:** `ML/data/data.json`

```bash
python ML/dataprocessing/merge.py
```

---

### 2. `encoding.py`

Converts text (categorical) columns into numbers using Label Encoding.

- **Reads:** `ML/data/data.json`
- **Writes:** `ML/data/data_encoded.json`

```bash
python ML/dataprocessing/encoding.py
```

Columns encoded: `Transaction Detail`, `Geological`, `Device Use`, `Gender`, `Location`, `Working Status`

---

### 3. `preprocessing.py`

Normalizes numeric columns and extracts new features from existing ones.

- **Reads:** `ML/data/data_encoded.json`
- **Writes:** `ML/data/data_processed.json`

```bash
python ML/dataprocessing/preprocessing.py
```

New features: `Age`, `Is_Weekend`, `Is_Night`, `Balance_to_Salary_Ratio`, `Tx_to_Balance_Ratio`

---

## Run Order

These scripts must be run **in order** before the ML pipeline:

```text
merge.py → encoding.py → preprocessing.py
```

Or run everything at once from the project root:

```bash
python run_pipeline.py
```
