import json
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "eta_best_model.pkl")
features = joblib.load(MODEL_DIR / "eta_best_features.pkl")

print("\n===== SELECTED ETA MODEL =====")
print("Model type:", type(model).__name__)

print("\n===== EXPECTED FEATURES =====")
print(features)

if hasattr(model, "feature_names_in_"):
    print("\nModel feature names:")
    print(list(model.feature_names_in_))

if hasattr(model, "n_features_in_"):
    print("\nNumber of features expected:", model.n_features_in_)

metadata_path = MODEL_DIR / "eta_best_model_metadata.json"

if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    print("\n===== MODEL METADATA =====")
    print(json.dumps(metadata, indent=2))

print("\nModel inspection completed.")