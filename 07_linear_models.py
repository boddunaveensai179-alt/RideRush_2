# ============================================================
# RIDERUSH - M2 LINEAR MODELS COMPARISON
#
# Models:
# 1. Linear Regression
# 2. Ridge Regression
# 3. Lasso Regression
# 4. Elastic Net
#
# Concepts:
# - StandardScaler
# - Pipeline
# - Regularization
# - Train/Test Split
# - MAE, RMSE, R2
# - Model Comparison
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("=" * 65)
print("RIDERUSH - LINEAR MODELS COMPARISON")
print("=" * 65)


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

data_path = "output/riderush_cleaned_data.parquet"

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Dataset not found: {data_path}. "
        "Run 01_preprocessing.py first."
    )

df = pd.read_parquet(data_path)

print("\nDataset loaded successfully.")
print("Rows:", len(df))


# ============================================================
# 2. SELECT FEATURES AND TARGET
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

target = "trip_duration"

required_columns = features + [target]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}. "
        "Check your preprocessing output."
    )

df = df[required_columns].copy()

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna()

X = df[features]
y = df[target]

print("\nTarget: Trip duration in minutes")
print("Features:", len(features))


# ============================================================
# 3. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Lasso Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.01, max_iter=10000))
    ]),

    "Elastic Net": Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000
        ))
    ])

}


# ============================================================
# 5. TRAIN AND EVALUATE
# ============================================================

results = []

trained_models = {}

os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

for name, model in models.items():

    print("\nTraining:", name)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
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

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2  :", round(r2, 4))


# ============================================================
# 6. DISPLAY COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.3f}".format,
            "RMSE": "{:.3f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_path = (
    "output/linear_models_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nComparison saved to:")
print(results_path)


# ============================================================
# 8. SAVE BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "models/best_eta_linear_pipeline.pkl"
)

joblib.dump(
    features,
    "models/eta_features.pkl"
)

print("\nBest model by MAE:", best_model_name)

print(
    "Saved model: models/best_eta_linear_pipeline.pkl"
)

print(
    "Saved features: models/eta_features.pkl"
)

print("\nM2 LINEAR MODELS EXPERIMENT COMPLETED.")