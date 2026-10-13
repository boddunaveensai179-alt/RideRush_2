import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model_paths = [
    BASE_DIR / "models" / "demand_model.pkl",
    BASE_DIR / "models" / "eta_model.pkl",
    BASE_DIR / "models" / "eta_rf_model.pkl",
    BASE_DIR / "models" / "eta_xgboost.pkl",
    BASE_DIR / "models" / "eta_lightgbm.pkl",
]

for path in model_paths:
    print("\n" + "=" * 55)

    if not path.exists():
        print("NOT FOUND:", path.name)
        continue

    model = joblib.load(path)

    print("Model:", path.name)
    print("Type:", type(model).__name__)
    print(
        "Expected feature count:",
        getattr(model, "n_features_in_", "Unknown")
    )
    print(
        "Training feature names:",
        getattr(model, "feature_names_in_", "Not available")
    )