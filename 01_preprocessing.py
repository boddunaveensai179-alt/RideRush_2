import os
import pandas as pd

os.makedirs("output", exist_ok=True)

FILE = "yellow_tripdata_2026-04.parquet"

print("=" * 55)
print("RIDErush - DATA PREPROCESSING")
print("=" * 55)

# 1. Load dataset
df = pd.read_parquet(FILE)

print("\n1. Dataset")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 2. Basic information
print("\n2. Important columns")
print(df.columns.tolist())

# 3. Missing values
missing = df.isnull().sum()
missing = missing[missing > 0]

print("\n3. Missing values")
print(missing)

# 4. Remove duplicates
df = df.drop_duplicates()

print("\n4. Duplicate removal")
print("Remaining rows:", len(df))

# 5. Datetime conversion
df["tpep_pickup_datetime"] = pd.to_datetime(
    df["tpep_pickup_datetime"],
    errors="coerce"
)

df["tpep_dropoff_datetime"] = pd.to_datetime(
    df["tpep_dropoff_datetime"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime"
    ]
)

# 6. Data cleaning
df["trip_duration"] = (
    df["tpep_dropoff_datetime"]
    - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60

df = df[
    (df["trip_duration"] > 0) &
    (df["trip_duration"] <= 180) &
    (df["trip_distance"] > 0) &
    (df["trip_distance"] <= 100) &
    (df["passenger_count"] > 0) &
    (df["passenger_count"] <= 8)
]

print("\n5. Data cleaning")
print("Clean rows:", len(df))

# 7. Feature engineering
df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
df["pickup_date"] = df["tpep_pickup_datetime"].dt.date

df["is_weekend"] = (
    df["pickup_dayofweek"] >= 5
).astype(int)

rush_hours = [7, 8, 9, 16, 17, 18, 19]

df["is_rush_hour"] = (
    df["pickup_hour"].isin(rush_hours)
).astype(int)

print("\n6. Features created")
print([
    "trip_duration",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour"
])

# 8. Save cleaned data
df.to_parquet(
    "output/riderush_cleaned_data.parquet",
    index=False
)

# 9. Create demand dataset
demand = (
    df.groupby([
        "pickup_date",
        "pickup_hour",
        "PULocationID"
    ])
    .size()
    .reset_index(name="demand")
)

demand["pickup_dayofweek"] = (
    pd.to_datetime(
        demand["pickup_date"]
    ).dt.dayofweek
)

demand["is_weekend"] = (
    demand["pickup_dayofweek"] >= 5
).astype(int)

demand["is_rush_hour"] = (
    demand["pickup_hour"].isin(rush_hours)
).astype(int)

demand.to_csv(
    "output/riderush_demand_data.csv",
    index=False
)

print("\n7. ML datasets created")
print("ETA dataset: riderush_cleaned_data.parquet")
print("Demand dataset: riderush_demand_data.csv")

print("\nPreprocessing completed.")