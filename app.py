"""
RideRush - Intelligent Taxi Demand and ETA Prediction System

Features:
1. Taxi-zone selection
2. Automatic zone-to-zone distance calculation
3. Taxi demand prediction
4. ETA prediction using the automatically selected model
5. Model performance information
6. Input validation and error handling
7. Health-check endpoint
"""

from pathlib import Path
import json
import math
import logging

import joblib
import pandas as pd

from flask import Flask, render_template, request, jsonify


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = BASE_DIR / "models"

# Demand prediction model
DEMAND_MODEL_PATH = MODEL_DIR / "demand_rf_model.pkl"

# Automatically selected ETA model
ETA_MODEL_PATH = MODEL_DIR / "eta_best_model.pkl"

# Selected ETA model metadata
ETA_METADATA_PATH = MODEL_DIR / "eta_best_model_metadata.json"

# Taxi-zone lookup
ZONE_LOOKUP_PATH = BASE_DIR / "taxi_zone_lookup.csv"

# Coordinates generated during ETA model preparation
COORDINATES_PATH = (
    BASE_DIR / "output" / "zone_coordinates_training.csv"
)

RUSH_HOURS = {7, 8, 9, 16, 17, 18, 19}


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

DEMAND_FEATURES = [
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "PULocationID",
]

ETA_FEATURES = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "PULocationID",
    "DOLocationID",
]


# ============================================================
# 3. LOAD MODELS
# ============================================================

def load_model(path):
    """Load a saved machine learning model."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required model not found: {path}"
        )

    return joblib.load(path)


demand_model = load_model(DEMAND_MODEL_PATH)
eta_model = load_model(ETA_MODEL_PATH)

logger.info(
    "Demand model loaded: %s",
    DEMAND_MODEL_PATH.name
)

logger.info(
    "Selected ETA model loaded: %s",
    ETA_MODEL_PATH.name
)


# ============================================================
# 4. LOAD ETA MODEL METADATA
# ============================================================

if not ETA_METADATA_PATH.exists():
    raise FileNotFoundError(
        f"ETA model metadata not found: {ETA_METADATA_PATH}"
    )

with open(ETA_METADATA_PATH, "r", encoding="utf-8") as file:
    eta_metadata = json.load(file)


SELECTED_ETA_MODEL = eta_metadata.get(
    "selected_model",
    type(eta_model).__name__
)

ETA_MAE = float(eta_metadata["mae_minutes"])
ETA_RMSE = float(eta_metadata["rmse_minutes"])
ETA_R2 = float(eta_metadata["r2"])

logger.info(
    "Selected ETA algorithm: %s",
    SELECTED_ETA_MODEL
)


# ============================================================
# 5. VERIFY MODEL FEATURE COMPATIBILITY
# ============================================================

def verify_model_features(model, expected_features, model_name):
    """
    Verify that the model accepts the features supplied by Flask.
    """

    actual_features = getattr(
        model,
        "feature_names_in_",
        None
    )

    expected_count = getattr(
        model,
        "n_features_in_",
        None
    )

    if (
        expected_count is not None
        and expected_count != len(expected_features)
    ):
        raise ValueError(
            f"{model_name} expects {expected_count} features, "
            f"but Flask supplies {len(expected_features)}."
        )

    if actual_features is not None:

        actual_features = list(actual_features)

        if actual_features != expected_features:
            raise ValueError(
                f"{model_name} feature mismatch.\n"
                f"Model features: {actual_features}\n"
                f"Application features: {expected_features}"
            )


verify_model_features(
    demand_model,
    DEMAND_FEATURES,
    "Demand model"
)

verify_model_features(
    eta_model,
    ETA_FEATURES,
    "Selected ETA model"
)

# Also verify the saved feature list against the selected model.
ETA_FEATURES_PATH = MODEL_DIR / "eta_best_features.pkl"

if not ETA_FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Selected model feature file not found: {ETA_FEATURES_PATH}"
    )

saved_eta_features = joblib.load(ETA_FEATURES_PATH)

if list(saved_eta_features) != ETA_FEATURES:
    raise ValueError(
        "Saved ETA features do not match Flask's ETA_FEATURES.\n"
        f"Saved: {list(saved_eta_features)}\n"
        f"Flask: {ETA_FEATURES}"
    )

logger.info("Model feature compatibility verified.")


# ============================================================
# 6. LOAD TAXI ZONES
# ============================================================

if not ZONE_LOOKUP_PATH.exists():
    raise FileNotFoundError(
        f"Taxi-zone lookup not found: {ZONE_LOOKUP_PATH}"
    )

zones = pd.read_csv(ZONE_LOOKUP_PATH)

required_zone_columns = {
    "LocationID",
    "Borough",
    "Zone",
}

if not required_zone_columns.issubset(zones.columns):
    raise ValueError(
        "taxi_zone_lookup.csv must contain "
        "LocationID, Borough, and Zone."
    )

zones = zones.dropna(
    subset=["LocationID", "Borough", "Zone"]
).copy()

zones["LocationID"] = zones["LocationID"].astype(int)

zone_dict = {
    int(row.LocationID): f"{row.Borough} - {row.Zone}"
    for row in zones.itertuples(index=False)
}

logger.info("Loaded %s taxi zones.", len(zone_dict))


# ============================================================
# 7. LOAD ZONE COORDINATES
# ============================================================

if not COORDINATES_PATH.exists():
    raise FileNotFoundError(
        "Zone coordinate file not found:\n"
        f"{COORDINATES_PATH}\n"
        "Run 06_eta_random_forest.py first."
    )

coordinates = pd.read_csv(COORDINATES_PATH)

required_coordinate_columns = {
    "LocationID",
    "latitude",
    "longitude",
}

if not required_coordinate_columns.issubset(coordinates.columns):
    raise ValueError(
        "Zone coordinate file must contain "
        "LocationID, latitude, and longitude."
    )

coordinates = coordinates.dropna(
    subset=["LocationID", "latitude", "longitude"]
).copy()

coordinate_dict = {}

for row in coordinates.itertuples(index=False):

    location_id = int(row.LocationID)
    latitude = float(row.latitude)
    longitude = float(row.longitude)

    if (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        coordinate_dict[location_id] = (
            latitude,
            longitude
        )

logger.info(
    "Loaded coordinates for %s taxi zones.",
    len(coordinate_dict)
)

if not coordinate_dict:
    raise ValueError(
        "No valid taxi-zone coordinates were loaded."
    )


# ============================================================
# 8. HAVERSINE DISTANCE
# ============================================================

def calculate_distance(pickup_id, drop_id):
    """
    Calculate straight-line distance between taxi-zone centroids.

    Returns miles, or None if coordinates are unavailable.
    """

    if (
        pickup_id not in coordinate_dict
        or drop_id not in coordinate_dict
    ):
        return None

    lat1, lon1 = coordinate_dict[pickup_id]
    lat2, lon2 = coordinate_dict[drop_id]

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    earth_radius_miles = 3958.8

    return round(
        earth_radius_miles * c,
        2
    )


# ============================================================
# 9. INPUT VALIDATION
# ============================================================

def validate_inputs(form):
    """Validate trip inputs received from the website."""

    try:
        pickup_location = int(form["pickup_location"])
        drop_location = int(form["drop_location"])
        passengers = int(form["passengers"])
        hour = int(form["hour"])
        day = int(form["day"])

    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "Please complete all trip fields with valid values."
        )

    if pickup_location not in zone_dict:
        raise ValueError("Invalid pickup location.")

    if drop_location not in zone_dict:
        raise ValueError("Invalid drop location.")

    if pickup_location == drop_location:
        raise ValueError(
            "Please choose different pickup and drop locations."
        )

    if not 1 <= passengers <= 8:
        raise ValueError(
            "Passenger count must be between 1 and 8."
        )

    if not 0 <= hour <= 23:
        raise ValueError(
            "Pickup hour must be between 0 and 23."
        )

    if not 0 <= day <= 6:
        raise ValueError(
            "Day of week must be between 0 and 6."
        )

    return {
        "pickup_location": pickup_location,
        "drop_location": drop_location,
        "passengers": passengers,
        "hour": hour,
        "day": day,
    }


# ============================================================
# 10. SHARED TEMPLATE CONTEXT
# ============================================================

def model_context():
    """
    Values available to the frontend for displaying
    the selected model and its recorded evaluation metrics.
    """

    return {
        "eta_model_name": SELECTED_ETA_MODEL,
        "eta_mae": ETA_MAE,
        "eta_rmse": ETA_RMSE,
        "eta_r2": ETA_R2,
    }


# ============================================================
# 11. HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        zones=zone_dict,
        **model_context()
    )


# ============================================================
# 12. AUTOMATIC DISTANCE API
# ============================================================

@app.route("/calculate_distance", methods=["POST"])
def calculate_distance_api():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "Invalid JSON request."
        }), 400

    try:
        pickup_id = int(data["pickup_location"])
        drop_id = int(data["drop_location"])

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "success": False,
            "message": "Invalid pickup or drop location."
        }), 400

    if (
        pickup_id not in zone_dict
        or drop_id not in zone_dict
    ):
        return jsonify({
            "success": False,
            "message": "Unknown taxi zone."
        }), 400

    if pickup_id == drop_id:
        return jsonify({
            "success": False,
            "message": "Select two different taxi zones."
        }), 400

    distance = calculate_distance(
        pickup_id,
        drop_id
    )

    if distance is None:
        return jsonify({
            "success": False,
            "message": "Coordinates unavailable for selected zones."
        }), 422

    if distance <= 0:
        return jsonify({
            "success": False,
            "message": "Calculated distance must be positive."
        }), 400

    return jsonify({
        "success": True,
        "distance": distance,
        "unit": "miles",
        "distance_type": "straight_line"
    })


# ============================================================
# 13. PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        inputs = validate_inputs(request.form)

        pickup_location = inputs["pickup_location"]
        drop_location = inputs["drop_location"]

        passengers = inputs["passengers"]
        hour = inputs["hour"]
        day = inputs["day"]

        is_weekend = int(day >= 5)
        is_rush_hour = int(hour in RUSH_HOURS)

        # ----------------------------------------------------
        # CALCULATE DISTANCE
        # ----------------------------------------------------

        distance = calculate_distance(
            pickup_location,
            drop_location
        )

        if distance is None:
            raise ValueError(
                "Distance could not be calculated for these zones."
            )

        if distance <= 0:
            raise ValueError(
                "Calculated distance must be positive. "
                "Please choose different locations."
            )

        # ----------------------------------------------------
        # DEMAND PREDICTION
        # ----------------------------------------------------

        demand_input = pd.DataFrame(
            [[
                hour,
                day,
                is_weekend,
                is_rush_hour,
                pickup_location,
            ]],
            columns=DEMAND_FEATURES
        )

        # ----------------------------------------------------
        # ETA PREDICTION
        # ----------------------------------------------------

        eta_input = pd.DataFrame(
            [[
                distance,
                passengers,
                hour,
                day,
                is_weekend,
                is_rush_hour,
                pickup_location,
                drop_location,
            ]],
            columns=ETA_FEATURES
        )

        # ----------------------------------------------------
        # GENERATE PREDICTIONS
        # ----------------------------------------------------

        predicted_demand = float(
            demand_model.predict(demand_input)[0]
        )

        predicted_eta = float(
            eta_model.predict(eta_input)[0]
        )

        if not (
            math.isfinite(predicted_demand)
            and math.isfinite(predicted_eta)
        ):
            raise ValueError(
                "The model returned an invalid prediction."
            )

        predicted_demand = round(
            max(0.0, predicted_demand),
            2
        )

        predicted_eta = round(
            max(0.0, predicted_eta),
            2
        )

        logger.info(
            "Prediction successful: pickup=%s, drop=%s, "
            "distance=%.2f, demand=%.2f, eta=%.2f",
            pickup_location,
            drop_location,
            distance,
            predicted_demand,
            predicted_eta
        )

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        return render_template(
            "index.html",
            zones=zone_dict,
            demand=predicted_demand,
            eta=predicted_eta,
            distance=distance,
            pickup_name=zone_dict[pickup_location],
            drop_name=zone_dict[drop_location],
            pickup_id=pickup_location,
            drop_id=drop_location,
            **model_context()
        )

    except ValueError as error:

        logger.warning(
            "Invalid prediction request: %s",
            error
        )

        return render_template(
            "index.html",
            zones=zone_dict,
            error=str(error),
            **model_context()
        ), 400

    except Exception:

        logger.exception(
            "Unexpected prediction error."
        )

        return render_template(
            "index.html",
            zones=zone_dict,
            error=(
                "Prediction failed. Please try again "
                "or check the application logs."
            ),
            **model_context()
        ), 500


# ============================================================
# 14. HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "application": "RideRush",
        "demand_model": DEMAND_MODEL_PATH.name,
        "eta_model": ETA_MODEL_PATH.name,
        "selected_eta_algorithm": SELECTED_ETA_MODEL,
        "eta_mae_minutes": ETA_MAE,
        "eta_rmse_minutes": ETA_RMSE,
        "eta_r2": ETA_R2,
        "available_zones": len(zone_dict),
        "available_coordinates": len(coordinate_dict),
    })


# ============================================================
# 15. START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )