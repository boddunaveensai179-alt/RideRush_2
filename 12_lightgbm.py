# ============================================================
# RIDERUSH
# ETA PREDICTION - LIGHTGBM
# ============================================================

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "output" / "riderush_cleaned_data.parquet"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("RIDErush - ETA LIGHTGBM")
print("=" * 55)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading dataset...", flush=True)

df = pd.read_parquet(DATA_PATH)

features = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "PULocationID",
    "DOLocationID"
]

target = "trip_duration"

df = df[features + [target]].replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

# Use the same sample size and random seed as XGBoost
SAMPLE_SIZE = 500_000

if len(df) > SAMPLE_SIZE:
    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    ).reset_index(drop=True)

print(f"Rows used: {len(df):,}", flush=True)

X = df[features]
y = df[target]

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")

# ============================================================
# 4. CREATE LIGHTGBM MODEL
# ============================================================

model = LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=8,
    verbosity=-1
)

# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining LightGBM...", flush=True)

model.fit(X_train, y_train)

print("Training completed!", flush=True)

# ============================================================
# 6. PREDICT
# ============================================================

predictions = model.predict(X_test)

# ============================================================
# 7. EVALUATE
# ============================================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n" + "=" * 55)
print("LIGHTGBM RESULTS")
print("=" * 55)

print(f"MAE : {mae:.4f} minutes")
print(f"RMSE: {rmse:.4f} minutes")
print(f"R2  : {r2:.4f}")

# ============================================================
# 8. SAVE MODEL AND FEATURES
# ============================================================

joblib.dump(
    model,
    MODEL_DIR / "eta_lightgbm.pkl"
)

joblib.dump(
    features,
    MODEL_DIR / "eta_lightgbm_features.pkl"
)

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results = pd.DataFrame([{
    "Model": "LightGBM",
    "Rows": len(df),
    "Estimators": 300,
    "Learning Rate": 0.05,
    "Max Depth": 8,
    "Num Leaves": 31,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}])

results.to_csv(
    OUTPUT_DIR / "lightgbm_results.csv",
    index=False
)

print("\nModel saved: models/eta_lightgbm.pkl")
print("Results saved: output/lightgbm_results.csv")

print("\nLightGBM completed.")