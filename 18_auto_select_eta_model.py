# ============================================================
# RIDERUSH - AUTOMATIC ETA MODEL SELECTION
# ============================================================

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. PATHS AND CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "output" / "riderush_cleaned_data.parquet"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "trip_duration"

FEATURES = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "PULocationID",
    "DOLocationID",
]


# ------------------------------------------------------------
# 2. LOAD AND PREPARE DATA
# ------------------------------------------------------------

def load_data():
    print("\n" + "=" * 65)
    print("RIDerush AUTOMATIC ETA MODEL SELECTION")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run 01_preprocessing.py first."
        )

    print("\nLoading cleaned taxi dataset...")

    df = pd.read_parquet(DATA_PATH)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert the model features and target to numeric values.
    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove invalid and infinite values.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_columns)

    # Keep valid trip durations and feature values.
    df = df[
        (df[TARGET] > 0)
        & (df[TARGET] <= 180)
        & (df["trip_distance"] > 0)
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 8)
        & (df["pickup_hour"].between(0, 23))
        & (df["pickup_dayofweek"].between(0, 6))
        & (df["is_weekend"].isin([0, 1]))
        & (df["is_rush_hour"].isin([0, 1]))
        & (df["PULocationID"] > 0)
        & (df["DOLocationID"] > 0)
    ].copy()

    # NYC taxi trip duration is measured in minutes.
    print(f"Valid rows: {len(df):,}")

    X = df[FEATURES]
    y = df[TARGET]

    return X, y


# ------------------------------------------------------------
# 3. DEFINE CANDIDATE MODELS
# ------------------------------------------------------------

def build_models():
    return {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=20,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=8,
            verbosity=0,
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=8,
            verbosity=-1,
        ),
    }


# ------------------------------------------------------------
# 4. TRAIN AND EVALUATE ALL MODELS
# ------------------------------------------------------------

def evaluate_models(X, y):
    print("\nCreating fixed training and test sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Training records: {len(X_train):,}")
    print(f"Test records:     {len(X_test):,}")

    models = build_models()

    results = []
    trained_models = {}

    for model_name, model in models.items():
        print("\n" + "-" * 60)
        print(f"Training: {model_name}")
        print("-" * 60)

        start_time = time.perf_counter()

        model.fit(X_train, y_train)

        training_seconds = time.perf_counter() - start_time

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        r2 = r2_score(y_test, predictions)

        results.append({
            "model": model_name,
            "mae_minutes": float(mae),
            "rmse_minutes": float(rmse),
            "r2": float(r2),
            "training_seconds": float(training_seconds),
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "features": ", ".join(FEATURES),
        })

        trained_models[model_name] = model

        print(f"MAE:              {mae:.4f} minutes")
        print(f"RMSE:             {rmse:.4f} minutes")
        print(f"R2:               {r2:.4f}")
        print(f"Training time:    {training_seconds:.2f} seconds")

    results_df = pd.DataFrame(results)

    # Primary selection criterion: lowest test MAE.
    # These test metrics are suitable for a preliminary comparison;
    # use a separate validation set for final model selection.
    results_df = results_df.sort_values(
        by=["mae_minutes", "rmse_minutes"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return results_df, trained_models


# ------------------------------------------------------------
# 5. SAVE THE WINNING MODEL
# ------------------------------------------------------------

def save_best_model(results_df, trained_models):
    best_row = results_df.iloc[0]
    best_name = best_row["model"]
    best_model = trained_models[best_name]

    model_path = MODEL_DIR / "eta_best_model.pkl"
    features_path = MODEL_DIR / "eta_best_features.pkl"
    metadata_path = MODEL_DIR / "eta_best_model_metadata.json"

    joblib.dump(best_model, model_path)
    joblib.dump(FEATURES, features_path)

    metadata = {
        "selected_model": best_name,
        "selection_metric": "lowest MAE",
        "mae_minutes": float(best_row["mae_minutes"]),
        "rmse_minutes": float(best_row["rmse_minutes"]),
        "r2": float(best_row["r2"]),
        "features": FEATURES,
        "target": TARGET,
        "target_unit": "minutes",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "dataset": DATA_PATH.name,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("BEST ETA MODEL SELECTED")
    print("=" * 65)

    print(f"Selected model: {best_name}")
    print(f"MAE:            {best_row['mae_minutes']:.4f} minutes")
    print(f"RMSE:           {best_row['rmse_minutes']:.4f} minutes")
    print(f"R2:             {best_row['r2']:.4f}")

    print(f"\nSaved model:    {model_path}")
    print(f"Saved features: {features_path}")
    print(f"Saved metadata: {metadata_path}")

    return best_name


# ------------------------------------------------------------
# 6. SAVE COMPARISON RESULTS
# ------------------------------------------------------------

def save_results(results_df):
    csv_path = OUTPUT_DIR / "eta_auto_model_comparison.csv"

    results_df.to_csv(csv_path, index=False)

    print("\nMODEL COMPARISON")
    print("=" * 90)

    print(
        results_df[
            [
                "model",
                "mae_minutes",
                "rmse_minutes",
                "r2",
                "training_seconds",
            ]
        ].to_string(
            index=False,
            formatters={
                "mae_minutes": "{:.4f}".format,
                "rmse_minutes": "{:.4f}".format,
                "r2": "{:.4f}".format,
                "training_seconds": "{:.2f}".format,
            },
        )
    )

    print(f"\nComparison saved to: {csv_path}")


# ------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------

def main():
    X, y = load_data()

    results_df, trained_models = evaluate_models(X, y)

    save_results(results_df)

    save_best_model(results_df, trained_models)

    print("\nAutomatic ETA model selection completed successfully.")


if __name__ == "__main__":
    main()