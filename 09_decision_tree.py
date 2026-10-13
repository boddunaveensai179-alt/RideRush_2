import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "output" / "riderush_cleaned_data.parquet"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Load data
df = pd.read_parquet(DATA_PATH)

# 3. Select features and target
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

X = df[features]
y = df[target]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 5. Create Decision Tree model
model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# 6. Train model
print("Training Decision Tree Regressor...")
model.fit(X_train, y_train)

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== DECISION TREE RESULTS ==========")
print(f"MAE  : {mae:.4f} minutes")
print(f"RMSE : {rmse:.4f} minutes")
print(f"R2   : {r2:.4f}")

# 9. Save model and feature names
joblib.dump(
    model,
    MODEL_DIR / "eta_decision_tree.pkl"
)

joblib.dump(
    features,
    MODEL_DIR / "eta_decision_tree_features.pkl"
)

# 10. Save results
results = pd.DataFrame([{
    "Model": "Decision Tree",
    "Max Depth": 10,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}])

results.to_csv(
    OUTPUT_DIR / "decision_tree_results.csv",
    index=False
)

print("\nModel saved to models/eta_decision_tree.pkl")
print("Results saved to output/decision_tree_results.csv")