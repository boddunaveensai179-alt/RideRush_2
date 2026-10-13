# ============================================================
# RIDERUSH
# ETA PREDICTION - LINEAR REGRESSION
# ============================================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PROJECT TITLE
# ============================================================

print("=" * 55)
print("RIDErush - ETA LINEAR REGRESSION")
print("=" * 55)


# ============================================================
# 2. LOAD CLEANED DATA
# ============================================================

df = pd.read_parquet(
    "output/riderush_cleaned_data.parquet"
)

print("\nDataset loaded successfully.")
print("Rows:", len(df))


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

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

X = df[features]

y = df["trip_duration"]


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 5. CREATE AND TRAIN MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

print("\nModel training completed.")


# ============================================================
# 6. SAVE TRAINED MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/eta_model.pkl"
)

print("Model saved:")
print("models/eta_model.pkl")


# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 9. DISPLAY IMPORTANT RESULTS
# ============================================================

print("\n" + "=" * 55)
print("ETA PREDICTION RESULTS")
print("=" * 55)

print("MAE :", round(mae, 2), "minutes")
print("RMSE:", round(rmse, 2), "minutes")
print("R2  :", round(r2, 4))


# ============================================================
# 10. COMPLETION
# ============================================================

print("\nETA Linear Regression completed.")
print("Ready for Flask.")