import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

PARQUET_FILE = OUTPUT_DIR / "riderush_cleaned_data.parquet"
DEMAND_CSV = OUTPUT_DIR / "riderush_demand_data.csv"
STATISTICS_CSV = OUTPUT_DIR / "riderush_statistics.csv"


def display_dataset(file_path, file_type):
    print("\n" + "=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    try:
        if file_type == "parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        print("\nDataset shape:")
        print(f"Rows: {df.shape[0]:,}")
        print(f"Columns: {df.shape[1]}")

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst 10 records:")
        print(df.head(10).to_string(index=False))

        print("\nMissing values:")
        print(df.isnull().sum().to_string())

    except Exception as error:
        print(f"Error reading file: {error}")


# Display cleaned taxi data
display_dataset(PARQUET_FILE, "parquet")

# Display demand data
display_dataset(DEMAND_CSV, "csv")

# Display statistics
display_dataset(STATISTICS_CSV, "csv")