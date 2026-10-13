import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("RIDErush - DEMAND LINEAR REGRESSION")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "output/riderush_demand_data.csv"
)

print("Dataset loaded:", df.shape)

# ============================================================
# 2. FEATURES
# ============================================================

features = [
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "PULocationID"
]

target = "demand"

X = df[features]
y = df[target]

# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ============================================================
# 4. TRAIN LINEAR REGRESSION
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# ============================================================
# 5. PREDICTION
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# 6. EVALUATION
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

print("\n" + "=" * 60)
print("DEMAND MODEL RESULTS")
print("=" * 60)

print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R²   :", round(r2, 4))

# ============================================================
# 7. SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/demand_model.pkl"
)

print("\nModel saved:")
print("models/demand_model.pkl")

# ============================================================
# 8. SAVE FEATURE INFORMATION
# ============================================================

joblib.dump(
    features,
    "models/demand_features.pkl"
)

print("Feature list saved:")
print("models/demand_features.pkl")

print("\nDemand Linear Regression completed.")