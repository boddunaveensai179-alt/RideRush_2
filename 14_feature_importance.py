# ============================================================
# RIDERUSH - FEATURE IMPORTANCE
# ============================================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "output" / "riderush_cleaned_data.parquet"
MODEL_PATH = BASE_DIR / "models" / "eta_xgboost.pkl"
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = BASE_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATA AND MODEL
# ------------------------------------------------------------

print("Loading dataset and XGBoost model...")

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
    [float("inf"), float("-inf")],
    float("nan")
).dropna()

# Use the same sample and split as the XGBoost experiment
df = df.sample(
    n=min(500_000, len(df)),
    random_state=42
).reset_index(drop=True)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------
# 3. BUILT-IN FEATURE IMPORTANCE
# ------------------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

print("\n========== XGBOOST FEATURE IMPORTANCE ==========")
print(importance_df.to_string(index=False))

# ------------------------------------------------------------
# 4. VISUALIZE IMPORTANCE
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("RideRush - XGBoost Feature Importance")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "xgboost_feature_importance.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 5. PERMUTATION IMPORTANCE
# ------------------------------------------------------------

# Use a smaller sample to keep this experiment efficient
X_eval = X_test.sample(
    n=min(10_000, len(X_test)),
    random_state=42
)

y_eval = y_test.loc[X_eval.index]

print("\nCalculating permutation importance...")

permutation = permutation_importance(
    model,
    X_eval,
    y_eval,
    scoring="neg_mean_absolute_error",
    n_repeats=3,
    random_state=42,
    n_jobs=4
)

permutation_df = pd.DataFrame({
    "Feature": features,
    "Importance_MAE_Increase": permutation.importances_mean
})

permutation_df = permutation_df.sort_values(
    "Importance_MAE_Increase",
    ascending=False
)

permutation_df.to_csv(
    OUTPUT_DIR / "permutation_importance.csv",
    index=False
)

print("\n========== PERMUTATION IMPORTANCE ==========")
print(permutation_df.to_string(index=False))

print("\nSaved:")
print("output/feature_importance.csv")
print("output/permutation_importance.csv")
print("plots/xgboost_feature_importance.png")