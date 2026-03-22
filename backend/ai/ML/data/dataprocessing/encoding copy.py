import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Columns to encode (text → numbers)
TEXT_COLUMNS = [
    'Transaction Detail',
    'Geological',
    'Device Use',
    'Gender',
    'Location',
    'Working Status',
]


def encode_columns(df: pd.DataFrame, columns: list) -> tuple:
    """
    Fit a LabelEncoder on each column and transform the values.
    Returns the transformed df and a dict of fitted encoders.
    """
    encoders = {}
    for col in columns:
        if col in df.columns:
            enc = LabelEncoder()
            df[col]       = enc.fit_transform(df[col].astype(str))
            encoders[col] = enc
    return df, encoders


def encode_and_export(input_path: str, output_path: str, encoders_path: str) -> pd.DataFrame:
    print(f"Loading data from: {input_path}")
    df = pd.read_json(input_path)
    print(f"Loaded {len(df)} rows.")
    print(f"Encoding columns: {TEXT_COLUMNS}")

    df, encoders = encode_columns(df, TEXT_COLUMNS)

    # Save encoders, required so inference uses the exact same mapping
    joblib.dump(encoders, encoders_path)
    print(f"Encoders saved to: {encoders_path}")

    df.to_json(output_path, orient='records', indent=4, force_ascii=False)
    print(f"Saved encoded data to: {output_path}")

    # Print the learned mapping for each column so you can verify
    for col, enc in encoders.items():
        mapping = dict(zip(enc.classes_, enc.transform(enc.classes_)))
        print(f"  {col}: {mapping}")

    return df


if __name__ == "__main__":
    HERE          = os.path.dirname(__file__)
    INPUT_FILE    = os.path.join(HERE, '..', 'data_2', 'data.json')
    OUTPUT_FILE   = os.path.join(HERE, '..', 'data_2', 'data_encoded.json')
    ENCODERS_PATH = os.path.join(HERE, '..', '..', 'models', 'encoders.pkl')

    encode_and_export(INPUT_FILE, OUTPUT_FILE, ENCODERS_PATH)