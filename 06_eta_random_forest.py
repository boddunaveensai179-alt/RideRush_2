# ============================================================
# RIDERUSH
# ETA PREDICTION - RANDOM FOREST
# Consistent zone-centroid distance for training and serving
# ============================================================

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

ZONE_LOOKUP_PATH = BASE_DIR / "taxi_zone_lookup.csv"
SHAPEFILE_PATH = BASE_DIR / "taxi_zones" / "taxi_zones.shp"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("RIDErush - ETA RANDOM FOREST")
print("=" * 65)

# ============================================================
# 2. VALIDATE INPUT FILES
# ============================================================

for path in [
    DATA_PATH,
    ZONE_LOOKUP_PATH,
    SHAPEFILE_PATH
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

# ============================================================
# 3. LOAD TAXI ZONE GEOMETRIES
# ============================================================

print("\nLoading taxi-zone geometries...", flush=True)

zones_gdf = gpd.read_file(SHAPEFILE_PATH)

if zones_gdf.crs is None:
    raise ValueError(
        "Taxi-zone shapefile has no CRS. "
        "Check the shapefile's .prj file."
    )

# Use a projected CRS for centroid calculations.
zones_projected = zones_gdf.to_crs(epsg=32618)

zones_projected["centroid"] = (
    zones_projected.geometry.centroid
)

centroids = zones_projected[
    ["LocationID", "centroid"]
].copy()

# Convert centroids to geographic coordinates.
centroids = centroids.set_geometry("centroid").to_crs(
    epsg=4326
)

centroids["longitude"] = centroids.geometry.x
centroids["latitude"] = centroids.geometry.y

coordinates = centroids[
    ["LocationID", "latitude", "longitude"]
].copy()

coordinates["LocationID"] = coordinates[
    "LocationID"
].astype(int)

coordinates = coordinates.dropna(
    subset=["LocationID", "latitude", "longitude"]
)

# Save the exact coordinate reference used for training.
coordinates_path = OUTPUT_DIR / "zone_coordinates_training.csv"

coordinates.to_csv(
    coordinates_path,
    index=False
)

coordinate_dict = {
    int(row.LocationID): (
        float(row.latitude),
        float(row.longitude)
    )
    for row in coordinates.itertuples(index=False)
}

print(
    f"Zone centroids loaded: {len(coordinate_dict)}",
    flush=True
)

# ============================================================
# 4. DISTANCE FUNCTION
# ============================================================

def haversine_distance(pickup_id, drop_id):

    if pickup_id not in coordinate_dict:
        return np.nan

    if drop_id not in coordinate_dict:
        return np.nan

    lat1, lon1 = coordinate_dict[pickup_id]
    lat2, lon2 = coordinate_dict[drop_id]

    lat1, lon1, lat2, lon2 = np.radians(
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return 3958.8 * c

# ============================================================
# 5. LOAD DATA
# ============================================================

print("\nLoading cleaned taxi dataset...", flush=True)

df = pd.read_parquet(DATA_PATH)

print(f"Original rows: {len(df):,}", flush=True)

# ============================================================
# 6. VALIDATE FEATURES
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
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in cleaned dataset: {missing_columns}"
    )

df = df[required_columns].copy()

# ============================================================
# 7. CREATE SERVING-COMPATIBLE DISTANCE
# ============================================================

print(
    "\nCalculating zone-centroid distance...",
    flush=True
)

df["PULocationID"] = pd.to_numeric(
    df["PULocationID"],
    errors="coerce"
)

df["DOLocationID"] = pd.to_numeric(
    df["DOLocationID"],
    errors="coerce"
)

# Remove invalid location IDs before integer conversion.
df = df.dropna(
    subset=["PULocationID", "DOLocationID"]
).copy()

df["PULocationID"] = df["PULocationID"].astype(int)
df["DOLocationID"] = df["DOLocationID"].astype(int)

df["trip_distance"] = [
    haversine_distance(pickup, drop)
    for pickup, drop in zip(
        df["PULocationID"],
        df["DOLocationID"]
    )
]

# ============================================================
# 8. CLEAN DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

# Remove impossible or unhelpful values.
df = df[
    (df["trip_distance"] > 0)
    & (df["passenger_count"].between(1, 8))
    & (df["pickup_hour"].between(0, 23))
    & (df["pickup_dayofweek"].between(0, 6))
    & (df["trip_duration"] > 0)
    & (df["trip_duration"] <= 180)
].copy()

print(
    f"Valid rows after cleaning: {len(df):,}",
    flush=True
)

# ============================================================
# 9. SAMPLE DATA
# ============================================================

SAMPLE_SIZE = 500_000

if len(df) > SAMPLE_SIZE:
    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    ).reset_index(drop=True)

print(f"Rows used for training: {len(df):,}", flush=True)

if len(df) < 100:
    raise ValueError(
        "Too few valid rows to train the ETA model."
    )

X = df[features]
y = df[target]

# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training rows: {len(X_train):,}", flush=True)
print(f"Testing rows: {len(X_test):,}", flush=True)

# ============================================================
# 11. CREATE RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# ============================================================
# 12. TRAIN
# ============================================================

print("\nTraining Random Forest...", flush=True)

model.fit(X_train, y_train)

print("\nTraining completed!", flush=True)

# ============================================================
# 13. EVALUATE
# ============================================================

print("\nGenerating test predictions...", flush=True)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 65)
print("ETA RANDOM FOREST RESULTS")
print("=" * 65)

print(f"MAE : {mae:.4f} minutes")
print(f"RMSE: {rmse:.4f} minutes")
print(f"R²  : {r2:.4f}")

# ============================================================
# 14. SAVE MODEL AND FEATURES
# ============================================================

model_path = MODEL_DIR / "eta_rf_model.pkl"

features_path = MODEL_DIR / "eta_rf_features.pkl"

joblib.dump(model, model_path)
joblib.dump(features, features_path)

# Save metrics for later comparison.
metrics = pd.DataFrame([{
    "model": "Random Forest",
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "rows": len(df),
    "distance_definition": "zone_centroid_haversine"
}])

metrics_path = OUTPUT_DIR / "eta_rf_updated_metrics.csv"

metrics.to_csv(metrics_path, index=False)

print("\nSaved:")
print(f"- {model_path}")
print(f"- {features_path}")
print(f"- {coordinates_path}")
print(f"- {metrics_path}")

print("\nETA Random Forest retraining completed.")