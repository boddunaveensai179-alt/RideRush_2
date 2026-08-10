# ============================================================
# RIDERUSH
# NYC TAXI TRIP DEMAND + ETA PREDICTION
# DATA LOADING + CLEANING + FEATURE ENGINEERING + EDA
# ============================================================

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ============================================================
# 1. CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("plots", exist_ok=True)
os.makedirs("output", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

FILE_PATH = "yellow_tripdata_2026-04.parquet"

print("=" * 70)
print("RIDERUSH - NYC TAXI DATA ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_parquet(FILE_PATH)

print("Dataset loaded successfully!")


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe(include="all"))


# ============================================================
# 4. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

print("Rows after duplicate removal:", len(df))


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing = df.isnull().sum()

missing_percentage = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": missing_percentage
})

missing_table = missing_table.sort_values(
    by="Percentage",
    ascending=False
)

print(missing_table)


# ============================================================
# 6. CONVERT DATETIME COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("DATETIME CONVERSION")
print("=" * 70)

df["tpep_pickup_datetime"] = pd.to_datetime(
    df["tpep_pickup_datetime"],
    errors="coerce"
)

df["tpep_dropoff_datetime"] = pd.to_datetime(
    df["tpep_dropoff_datetime"],
    errors="coerce"
)

print("Datetime conversion completed.")


# ============================================================
# 7. REMOVE INVALID DATETIME RECORDS
# ============================================================

before = len(df)

df = df.dropna(
    subset=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime"
    ]
)

after = len(df)

print("Invalid datetime rows removed:", before - after)


# ============================================================
# 8. CREATE TRIP DURATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRIP DURATION")
print("=" * 70)

df["trip_duration"] = (
    df["tpep_dropoff_datetime"]
    - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60

print("\nTrip duration statistics:")
print(df["trip_duration"].describe())


# ============================================================
# 9. REMOVE INVALID TRIP DURATIONS
# ============================================================

before = len(df)

df = df[
    (df["trip_duration"] > 0) &
    (df["trip_duration"] <= 180)
]

after = len(df)

print("\nInvalid trip durations removed:", before - after)


# ============================================================
# 10. REMOVE INVALID TRIP DISTANCES
# ============================================================

print("\n" + "=" * 70)
print("TRIP DISTANCE CLEANING")
print("=" * 70)

before = len(df)

df = df[
    (df["trip_distance"] > 0) &
    (df["trip_distance"] <= 100)
]

after = len(df)

print("Invalid distance rows removed:", before - after)


# ============================================================
# 11. CLEAN PASSENGER COUNT
# ============================================================

print("\n" + "=" * 70)
print("PASSENGER COUNT CLEANING")
print("=" * 70)

before = len(df)

df = df[
    (df["passenger_count"] > 0) &
    (df["passenger_count"] <= 8)
]

after = len(df)

print("Invalid passenger rows removed:", before - after)


# ============================================================
# 12. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# Hour
df["pickup_hour"] = (
    df["tpep_pickup_datetime"].dt.hour
)

# Day
df["pickup_day"] = (
    df["tpep_pickup_datetime"].dt.day
)

# Month
df["pickup_month"] = (
    df["tpep_pickup_datetime"].dt.month
)

# Day of week
df["pickup_dayofweek"] = (
    df["tpep_pickup_datetime"].dt.dayofweek
)

# Weekend
df["is_weekend"] = (
    df["pickup_dayofweek"] >= 5
).astype(int)

# Rush hour
rush_hours = [7, 8, 9, 16, 17, 18, 19]

df["is_rush_hour"] = (
    df["pickup_hour"].isin(rush_hours)
).astype(int)

# Date
df["pickup_date"] = (
    df["tpep_pickup_datetime"].dt.date
)

print("Feature engineering completed.")

print("\nNew features:")
print([
    "trip_duration",
    "pickup_hour",
    "pickup_day",
    "pickup_month",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "pickup_date"
])


# ============================================================
# 13. FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nFinal columns:")
print(df.columns.tolist())


# ============================================================
# 14. EDA - TOTAL TRIPS
# ============================================================

print("\n" + "=" * 70)
print("EDA")
print("=" * 70)

print("\nTotal valid trips:", len(df))


# ============================================================
# 15. EDA - TRIP DURATION DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["trip_duration"],
    bins=50,
    kde=True
)

plt.title("RideRush - Trip Duration Distribution")
plt.xlabel("Trip Duration (minutes)")
plt.ylabel("Number of Trips")

plt.tight_layout()

plt.savefig(
    "plots/01_trip_duration_distribution.png"
)

plt.show()


# ============================================================
# 16. EDA - TRIP DISTANCE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["trip_distance"],
    bins=50,
    kde=True
)

plt.title("RideRush - Trip Distance Distribution")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Number of Trips")

plt.tight_layout()

plt.savefig(
    "plots/02_trip_distance_distribution.png"
)

plt.show()


# ============================================================
# 17. EDA - DEMAND BY HOUR
# ============================================================

hourly_demand = (
    df.groupby("pickup_hour")
    .size()
)

print("\nTrips by hour:")
print(hourly_demand)

plt.figure(figsize=(12, 6))

hourly_demand.plot(kind="bar")

plt.title("RideRush - Taxi Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "plots/03_demand_by_hour.png"
)

plt.show()


# ============================================================
# 18. EDA - DEMAND BY DAY OF WEEK
# ============================================================

daily_demand = (
    df.groupby("pickup_dayofweek")
    .size()
)

day_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

plt.figure(figsize=(10, 6))

plt.bar(
    day_names,
    daily_demand.reindex(range(7))
)

plt.title("RideRush - Taxi Demand by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Trips")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "plots/04_demand_by_day.png"
)

plt.show()


# ============================================================
# 19. EDA - RUSH HOUR
# ============================================================

rush_demand = (
    df.groupby("is_rush_hour")
    .size()
)

print("\nRush hour analysis:")
print(rush_demand)

plt.figure(figsize=(8, 6))

plt.bar(
    ["Normal Hours", "Rush Hours"],
    [
        rush_demand.get(0, 0),
        rush_demand.get(1, 0)
    ]
)

plt.title("RideRush - Rush Hour vs Normal Hour")
plt.xlabel("Time Category")
plt.ylabel("Number of Trips")

plt.tight_layout()

plt.savefig(
    "plots/05_rush_hour.png"
)

plt.show()


# ============================================================
# 20. EDA - PASSENGER COUNT
# ============================================================

passenger_counts = (
    df["passenger_count"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 6))

plt.bar(
    passenger_counts.index.astype(str),
    passenger_counts.values
)

plt.title("RideRush - Passenger Count Distribution")
plt.xlabel("Number of Passengers")
plt.ylabel("Number of Trips")

plt.tight_layout()

plt.savefig(
    "plots/06_passenger_count.png"
)

plt.show()


# ============================================================
# 21. EDA - DISTANCE VS TRIP DURATION
# ============================================================

sample_size = min(50000, len(df))

sample_df = df.sample(
    sample_size,
    random_state=42
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sample_df,
    x="trip_distance",
    y="trip_duration",
    alpha=0.3
)

plt.title("RideRush - Trip Distance vs Trip Duration")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Trip Duration (minutes)")

plt.tight_layout()

plt.savefig(
    "plots/07_distance_vs_duration.png"
)

plt.show()


# ============================================================
# 22. EDA - RUSH HOUR VS TRIP DURATION
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=sample_df,
    x="is_rush_hour",
    y="trip_duration"
)

plt.title("RideRush - Trip Duration: Rush Hour vs Normal")
plt.xlabel("Rush Hour (0 = No, 1 = Yes)")
plt.ylabel("Trip Duration (minutes)")

plt.tight_layout()

plt.savefig(
    "plots/08_rush_vs_duration.png"
)

plt.show()


# ============================================================
# 23. EDA - CORRELATION MATRIX
# ============================================================

numeric_columns = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "trip_duration"
]

# Only use columns that actually exist
numeric_columns = [
    col for col in numeric_columns
    if col in df.columns
]

correlation = df[numeric_columns].corr()

print("\nCorrelation matrix:")
print(correlation)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("RideRush - Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "plots/09_correlation_matrix.png"
)

plt.show()


# ============================================================
# 24. EDA - TOP PICKUP LOCATIONS
# ============================================================

if "PULocationID" in df.columns:

    top_pickup = (
        df["PULocationID"]
        .value_counts()
        .head(20)
    )

    print("\nTop 20 pickup locations:")
    print(top_pickup)

    plt.figure(figsize=(12, 6))

    top_pickup.plot(kind="bar")

    plt.title("RideRush - Top 20 Pickup Locations")
    plt.xlabel("Pickup Location ID")
    plt.ylabel("Number of Trips")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "plots/10_top_pickup_locations.png"
    )

    plt.show()


# ============================================================
# 25. EDA - TOP DROP-OFF LOCATIONS
# ============================================================

if "DOLocationID" in df.columns:

    top_dropoff = (
        df["DOLocationID"]
        .value_counts()
        .head(20)
    )

    print("\nTop 20 drop-off locations:")
    print(top_dropoff)

    plt.figure(figsize=(12, 6))

    top_dropoff.plot(kind="bar")

    plt.title("RideRush - Top 20 Drop-off Locations")
    plt.xlabel("Drop-off Location ID")
    plt.ylabel("Number of Trips")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "plots/11_top_dropoff_locations.png"
    )

    plt.show()


# ============================================================
# 26. CREATE DEMAND DATASET
# ============================================================

print("\n" + "=" * 70)
print("CREATING DEMAND DATASET")
print("=" * 70)

demand_df = (
    df.groupby([
        "pickup_date",
        "pickup_hour",
        "PULocationID"
    ])
    .size()
    .reset_index(name="demand")
)

print("\nDemand dataset:")
print(demand_df.head())

print("\nDemand dataset shape:")
print(demand_df.shape)


# ============================================================
# 27. DEMAND BY LOCATION
# ============================================================

location_demand = (
    demand_df.groupby("PULocationID")["demand"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(12, 6))

location_demand.plot(kind="bar")

plt.title("RideRush - Highest Demand Pickup Locations")
plt.xlabel("Pickup Location ID")
plt.ylabel("Total Demand")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "plots/12_highest_demand_locations.png"
)

plt.show()


# ============================================================
# 28. DEMAND HEATMAP - HOUR VS DAY
# ============================================================

hour_day_demand = (
    df.groupby([
        "pickup_dayofweek",
        "pickup_hour"
    ])
    .size()
    .unstack(fill_value=0)
)

plt.figure(figsize=(14, 7))

sns.heatmap(
    hour_day_demand,
    cmap="YlOrRd"
)

plt.title("RideRush - Taxi Demand by Day and Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")

plt.tight_layout()

plt.savefig(
    "plots/13_demand_heatmap.png"
)

plt.show()


# ============================================================
# 29. SAVE CLEANED DATASET
# ============================================================

print("\n" + "=" * 70)
print("SAVING OUTPUT")
print("=" * 70)

cleaned_file = "output/riderush_cleaned_data.parquet"

df.to_parquet(
    cleaned_file,
    index=False
)

print("Cleaned dataset saved:")
print(cleaned_file)


# ============================================================
# 30. SAVE DEMAND DATASET
# ============================================================

demand_file = "output/riderush_demand_data.csv"

demand_df.to_csv(
    demand_file,
    index=False
)

print("Demand dataset saved:")
print(demand_file)


# ============================================================
# 31. SAVE BASIC STATISTICS
# ============================================================

statistics_file = "output/riderush_statistics.csv"

df.describe().to_csv(
    statistics_file
)

print("Statistics saved:")
print(statistics_file)


# ============================================================
# 32. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RIDERUSH EDA COMPLETED")
print("=" * 70)

print("\nFinal number of trips:", len(df))

print("\nFinal number of features:", len(df.columns))

print("\nOutput files:")
print("1. output/riderush_cleaned_data.parquet")
print("2. output/riderush_demand_data.csv")
print("3. output/riderush_statistics.csv")

print("\nEDA plots:")
print("Check the 'plots' folder.")

print("\nNext step:")
print("Build Demand Prediction and ETA Prediction ML models.")

print("\nRideRush pipeline completed successfully!")