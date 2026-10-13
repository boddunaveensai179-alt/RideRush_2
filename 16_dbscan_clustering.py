"""
RideRush - DBSCAN Clustering
----------------------------
Purpose:
1. Identify groups in taxi demand observations.
2. Detect unusual observations as noise.
3. Tune DBSCAN parameters.
4. Evaluate clustering using silhouette score.
5. Save cluster results, summaries, and visualization.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=UserWarning)

# =======================================================
# 1. CONFIGURATION
# =======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "output",
    "riderush_demand_data.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

SAMPLE_SIZE = 20_000
RANDOM_STATE = 42

# Parameters to test
EPS_VALUES = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
MIN_SAMPLES_VALUES = [5, 10, 20, 30]

SILHOUETTE_SAMPLE_SIZE = 3000

# =======================================================
# 2. LOAD DATA
# =======================================================

print("=" * 65)
print("RIDErush - DBSCAN CLUSTERING")
print("=" * 65)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Demand dataset not found:\n{INPUT_FILE}\n"
        "Run 01_preprocessing.py first."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = ["pickup_hour", "demand"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Keep only the features used for clustering
df = df[required_columns].copy()

# Convert values to numeric and remove invalid observations
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

if len(df) < 3:
    raise ValueError(
        "Not enough valid observations for clustering."
    )

# =======================================================
# 3. SAMPLE DATA
# =======================================================

if len(df) > SAMPLE_SIZE:
    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).copy()

df.reset_index(drop=True, inplace=True)

print(f"\nRows used: {len(df):,}")
print("Features used: pickup_hour, demand")

# =======================================================
# 4. FEATURE SCALING
# =======================================================

# Scaling is important because demand and pickup_hour
# have different numerical ranges.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[["pickup_hour", "demand"]]
)

print("Feature scaling completed.")

# =======================================================
# 5. DBSCAN PARAMETER TUNING
# =======================================================

print("\n" + "=" * 65)
print("DBSCAN PARAMETER TUNING")
print("=" * 65)

results = []

best_score = -1.0
best_labels = None
best_params = None

for eps in EPS_VALUES:
    for min_samples in MIN_SAMPLES_VALUES:

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean"
        )

        labels = model.fit_predict(X_scaled)

        # DBSCAN uses -1 to represent noise.
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]

        cluster_count = len(set(valid_labels))
        noise_count = int(np.sum(labels == -1))
        noise_percentage = noise_count / len(labels) * 100

        score = np.nan

        # Silhouette requires at least two clusters.
        if (
            cluster_count >= 2
            and cluster_count < int(valid_mask.sum())
        ):
            try:
                score = silhouette_score(
                    X_scaled[valid_mask],
                    valid_labels,
                    sample_size=min(
                        SILHOUETTE_SAMPLE_SIZE,
                        int(valid_mask.sum())
                    ),
                    random_state=RANDOM_STATE
                )
            except ValueError:
                score = np.nan

        results.append({
            "eps": eps,
            "min_samples": min_samples,
            "clusters": cluster_count,
            "noise_count": noise_count,
            "noise_percentage": noise_percentage,
            "silhouette_score": score
        })

        score_text = (
            f"{score:.4f}"
            if np.isfinite(score)
            else "Unavailable"
        )

        print(
            f"eps={eps:<4} | "
            f"min_samples={min_samples:<2} | "
            f"clusters={cluster_count:<3} | "
            f"noise={noise_percentage:>6.2f}% | "
            f"silhouette={score_text}"
        )

        # Choose the best valid silhouette score.
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_labels = labels.copy()
            best_params = {
                "eps": eps,
                "min_samples": min_samples
            }

# =======================================================
# 6. SAVE PARAMETER TUNING RESULTS
# =======================================================

tuning_df = pd.DataFrame(results)

tuning_file = os.path.join(
    OUTPUT_DIR,
    "dbscan_parameter_results.csv"
)

tuning_df.to_csv(tuning_file, index=False)

print(f"\nParameter tuning results saved to:\n{tuning_file}")

# =======================================================
# 7. CHECK FOR A VALID CLUSTERING RESULT
# =======================================================

if best_labels is None:
    print("\n" + "=" * 65)
    print("NO VALID MULTI-CLUSTER SOLUTION FOUND")
    print("=" * 65)

    print(
        "None of the tested parameter combinations produced "
        "a valid silhouette score."
    )

    print("\nTry:")
    print("1. Expanding the EPS_VALUES range.")
    print("2. Examining the distribution of demand.")
    print("3. Adding meaningful features such as day of week.")
    print("4. Testing a different clustering algorithm.")

    # Save tuning results, but do not create misleading clusters.
    raise SystemExit(
        "DBSCAN tuning completed without a valid multi-cluster result."
    )

# =======================================================
# 8. APPLY THE BEST PARAMETERS
# =======================================================

print("\n" + "=" * 65)
print("BEST DBSCAN RESULT")
print("=" * 65)

print(f"Best eps: {best_params['eps']}")
print(f"Best min_samples: {best_params['min_samples']}")
print(f"Silhouette score: {best_score:.4f}")

df["cluster"] = best_labels

cluster_count = len(set(best_labels) - {-1})
noise_count = int(np.sum(best_labels == -1))
noise_percentage = noise_count / len(best_labels) * 100

print(f"Rows analyzed: {len(df):,}")
print(f"Clusters found: {cluster_count}")
print(f"Noise observations: {noise_count:,}")
print(f"Noise percentage: {noise_percentage:.2f}%")

# =======================================================
# 9. CLUSTER SUMMARY
# =======================================================

print("\n" + "=" * 65)
print("CLUSTER SUMMARY")
print("=" * 65)

# Exclude noise from the cluster summary.
clustered_data = df[df["cluster"] != -1].copy()

summary = (
    clustered_data
    .groupby("cluster")
    .agg(
        average_pickup_hour=("pickup_hour", "mean"),
        average_demand=("demand", "mean"),
        minimum_demand=("demand", "min"),
        maximum_demand=("demand", "max"),
        records=("cluster", "size")
    )
    .reset_index()
)

if summary.empty:
    print("No non-noise clusters are available.")
else:
    print(summary.round(3).to_string(index=False))

# =======================================================
# 10. SAVE CLUSTER RESULTS
# =======================================================

clustered_output = os.path.join(
    OUTPUT_DIR,
    "dbscan_clustered_demand.csv"
)

summary_output = os.path.join(
    OUTPUT_DIR,
    "dbscan_cluster_summary.csv"
)

df.to_csv(clustered_output, index=False)
summary.to_csv(summary_output, index=False)

print("\nSaved cluster data:")
print(clustered_output)
print(summary_output)

# =======================================================
# 11. VISUALIZE CLUSTERS
# =======================================================

print("\nGenerating DBSCAN visualization...")

plt.figure(figsize=(12, 7))

scatter = plt.scatter(
    df["pickup_hour"],
    df["demand"],
    c=df["cluster"],
    s=12,
    alpha=0.65
)

plt.xlabel("Pickup Hour")
plt.ylabel("Taxi Demand")
plt.title(
    "RideRush: DBSCAN Clustering of Taxi Demand\n"
    f"eps={best_params['eps']}, "
    f"min_samples={best_params['min_samples']}, "
    f"silhouette={best_score:.3f}"
)

plt.colorbar(scatter, label="Cluster Label (-1 = Noise)")
plt.grid(True, alpha=0.25)
plt.tight_layout()

plot_file = os.path.join(
    PLOTS_DIR,
    "dbscan_clusters.png"
)

plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Cluster visualization saved to:\n{plot_file}")

# =======================================================
# 12. FINAL SUMMARY
# =======================================================

print("\n" + "=" * 65)
print("DBSCAN CLUSTERING COMPLETED")
print("=" * 65)

print(f"Best parameters: {best_params}")
print(f"Clusters found: {cluster_count}")
print(f"Noise percentage: {noise_percentage:.2f}%")
print(f"Silhouette score: {best_score:.4f}")

print("\nFiles generated:")
print("- output/dbscan_parameter_results.csv")
print("- output/dbscan_clustered_demand.csv")
print("- output/dbscan_cluster_summary.csv")
print("- plots/dbscan_clusters.png")

print("\nNext step: inspect the clusters and compare them")
print("with K-Means results.")