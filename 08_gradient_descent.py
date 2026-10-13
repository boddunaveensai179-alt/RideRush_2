import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load cleaned taxi data
DATA_PATH = "output/riderush_cleaned_data.parquet"

df = pd.read_parquet(DATA_PATH)

# 2. Select features and target
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

# 3. Prepare data
df = df[features + [target]].replace([np.inf, -np.inf], np.nan)
df = df.dropna()

X = df[features]
y = df[target]

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 5. Define models
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "SGD - Learning Rate 0.01": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SGDRegressor(
            loss="squared_error",
            learning_rate="constant",
            eta0=0.01,
            max_iter=2000,
            tol=1e-4,
            random_state=42
        ))
    ]),

    "SGD - Learning Rate 0.001": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SGDRegressor(
            loss="squared_error",
            learning_rate="constant",
            eta0=0.001,
            max_iter=2000,
            tol=1e-4,
            random_state=42
        ))
    ])
}

# 6. Train and evaluate
results = []

for name, model in models.items():
    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# 7. Display comparison
results_df = pd.DataFrame(results).sort_values("MAE")

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# 8. Save results
results_df.to_csv(
    "output/gradient_descent_comparison.csv",
    index=False
)

print("\nResults saved to output/gradient_descent_comparison.csv")