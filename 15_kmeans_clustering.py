# ============================================================
# RIDERUSH - K-MEANS DEMAND CLUSTERING
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "output" / "riderush_demand_data.csv"
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = BASE_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("RIDErush - K-MEANS DEMAND CLUSTERING")
print("=" * 55)

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded.")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# ============================================================
# 3. PREPARE FEATURES
# ============================================================

required = [
    "pickup_hour",
    "PULocationID",
    "demand"
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        "Check output/riderush_demand_data.csv."
    )

X = df[required].copy()

X = X.replace(
    [float("inf"), float("-inf")],
    float("nan")
).dropna()

# ============================================================
# 4. SCALE FEATURES
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures scaled successfully.")

# ============================================================
# 5. ELBOW METHOD
# ============================================================

inertias = []

k_values = range(2, 9)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("RideRush - K-Means Elbow Method")
plt.xticks(list(k_values))
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "kmeans_elbow.png",
    dpi=300
)

plt.close()

# ============================================================
# 6. TRAIN FINAL CLUSTERING MODEL
# ============================================================

# Start with K=4, then review the elbow plot.
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

X["cluster"] = cluster_labels

# ============================================================
# 7. EVALUATE CLUSTERING
# ============================================================

score = silhouette_score(
    X_scaled,
    cluster_labels,
    sample_size=min(10_000, len(X_scaled)),
    random_state=42
)

print("\n========== K-MEANS RESULTS ==========")
print("Number of clusters:", 4)
print(f"Silhouette score: {score:.4f}")

# ============================================================
# 8. SUMMARIZE CLUSTERS
# ============================================================

summary = X.groupby("cluster").agg(
    average_pickup_hour=("pickup_hour", "mean"),
    average_location_id=("PULocationID", "mean"),
    average_demand=("demand", "mean"),
    number_of_records=("demand", "size")
).reset_index()

print("\n========== CLUSTER SUMMARY ==========")
print(summary.to_string(index=False))

# ============================================================
# 9. SAVE RESULTS
# ============================================================

X.to_csv(
    OUTPUT_DIR / "kmeans_clustered_demand.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "kmeans_cluster_summary.csv",
    index=False
)

print("\nSaved:")
print("output/kmeans_clustered_demand.csv")
print("output/kmeans_cluster_summary.csv")
print("plots/kmeans_elbow.png")

print("\nK-Means clustering completed.")