import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_columns(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    encoders = {}
    for col in columns:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder

    return df, encoders


def encode_and_export(input_path, output_path, columns=None):
    # Step 1: Load data
    print(f"Loading data from: {input_path}")
    df = pd.read_json(input_path)
    print(f"Loaded {len(df)} rows.")

    # Step 2: Show which columns will be encoded
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Encoding columns: {columns}")

    # Step 3: Encode
    df, encoders = encode_columns(df, columns)

    # Step 4: Save
    df.to_json(output_path, orient='records', indent=4, force_ascii=False)
    print(f"Saved encoded data to: {output_path}")
    print("Done!")

    return df


# Run directly to encode the labeled dataset
if __name__ == "__main__":
    INPUT_FILE = 'ML/data/data_2/data.json'
    OUTPUT_FILE = 'ML/data/data_2/data_encoded.json'

    # Columns to encode (text → numbers)
    TEXT_COLUMNS = [
        'Transaction Detail', 'Geological', 'Device Use',
        'Gender', 'Location', 'Working Status'
    ]

    encode_and_export(INPUT_FILE, OUTPUT_FILE, columns=TEXT_COLUMNS)
