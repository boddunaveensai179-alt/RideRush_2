"""
RideRush - Demand Prediction using Random Forest

Target:
Aggregated taxi demand for a pickup date, hour,
and pickup zone, as defined by the preprocessing pipeline.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "output" / "riderush_demand_data.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "demand_rf_model.pkl"
FEATURES_PATH = MODEL_DIR / "demand_rf_features.pkl"
METRICS_PATH = OUTPUT_DIR / "demand_rf_metrics.csv"


# ============================================================
# 2. CONFIGURATION
# ============================================================

FEATURES = [
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "PULocationID"
]

TARGET = "demand"

print("=" * 60)
print("RIDErush - DEMAND RANDOM FOREST")
print("=" * 60)


# ============================================================
# 3. LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Demand dataset not found: {DATA_PATH}\n"
        "Run 01_preprocessing.py first."
    )

df = pd.read_csv(DATA_PATH)

print(f"\nOriginal rows: {len(df):,}")

required_columns = FEATURES + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 4. CLEAN DATA
# ============================================================

df = df[required_columns].copy()

for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

df.dropna(inplace=True)

# Demand counts should be nonnegative.
df = df[df[TARGET] >= 0].copy()

# Basic validation of the known input ranges.
df = df[
    df["pickup_hour"].between(0, 23)
    & df["pickup_dayofweek"].between(0, 6)
    & df["is_weekend"].isin([0, 1])
    & df["is_rush_hour"].isin([0, 1])
].copy()

df["PULocationID"] = df["PULocationID"].astype(int)

if len(df) < 100:
    raise ValueError(
        "Not enough valid demand observations for training."
    )

print(f"Valid rows: {len(df):,}")


# ============================================================
# 5. FEATURES AND TARGET
# ============================================================

X = df[FEATURES]
y = df[TARGET]

print("\nFeatures:")
print(FEATURES)

print("\nTarget:", TARGET)

print("\nDemand statistics:")
print(y.describe().round(2))


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")


# ============================================================
# 7. CREATE MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...", flush=True)

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 9. EVALUATE MODEL
# ============================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DEMAND RANDOM FOREST RESULTS")
print("=" * 60)

print(f"MAE : {mae:.4f} rides")
print(f"RMSE: {rmse:.4f} rides")
print(f"R²  : {r2:.4f}")


# ============================================================
# 11. SAVE MODEL AND FEATURES
# ============================================================

joblib.dump(model, MODEL_PATH)
joblib.dump(FEATURES, FEATURES_PATH)

print("\nModel saved:", MODEL_PATH)
print("Feature list saved:", FEATURES_PATH)


# ============================================================
# 12. SAVE METRICS
# ============================================================

metrics = pd.DataFrame([{
    "model": "Random Forest",
    "target": TARGET,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "training_rows": len(X_train),
    "testing_rows": len(X_test)
}])

metrics.to_csv(
    METRICS_PATH,
    index=False
)

print("Metrics saved:", METRICS_PATH)


# ============================================================
# 13. SAMPLE PREDICTIONS
# ============================================================

comparison = pd.DataFrame({
    "actual_demand": y_test.iloc[:10].values,
    "predicted_demand": y_pred[:10]
})

print("\nSample predictions:")
print(comparison.round(2).to_string(index=False))


print("\n" + "=" * 60)
print("DEMAND RANDOM FOREST COMPLETED")
print("=" * 60)