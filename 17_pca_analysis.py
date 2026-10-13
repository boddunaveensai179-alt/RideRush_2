"""
RideRush - Principal Component Analysis (PCA)

Purpose:
1. Reduce the dimensionality of taxi-demand data.
2. Calculate explained variance.
3. Determine the number of components needed
   to retain 95% of the variance.
4. Generate PCA visualizations.
5. Save results for project evaluation.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

RANDOM_STATE = 42
SAMPLE_SIZE = 20_000

FEATURES = [
    "pickup_hour",
    "PULocationID",
    "demand",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
]

# =======================================================
# 2. LOAD DATA
# =======================================================

print("=" * 65)
print("RIDErush - PCA DIMENSIONALITY REDUCTION")
print("=" * 65)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}\n"
        "Run 01_preprocessing.py first."
    )

df = pd.read_csv(INPUT_FILE)

available_features = [
    feature for feature in FEATURES
    if feature in df.columns
]

if len(available_features) < 2:
    raise ValueError(
        "PCA requires at least two available numerical features."
    )

print("\nAvailable features:")
print(available_features)

# =======================================================
# 3. PREPARE DATA
# =======================================================

data = df[available_features].copy()

for column in available_features:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

if len(data) < 3:
    raise ValueError("Not enough valid rows for PCA.")

if len(data) > SAMPLE_SIZE:
    data = data.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

print(f"\nRows used: {len(data):,}")
print(f"Number of features: {len(available_features)}")

# =======================================================
# 4. STANDARDIZE FEATURES
# =======================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(data)

print("\nFeature standardization completed.")

# =======================================================
# 5. FIT PCA
# =======================================================

pca = PCA(random_state=RANDOM_STATE)

X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Number of components required to retain 95% variance
components_95 = (
    int(np.searchsorted(cumulative_variance, 0.95)) + 1
)

print("\n" + "=" * 65)
print("PCA RESULTS")
print("=" * 65)

print(f"Original number of features: {X_scaled.shape[1]}")
print(f"Number of principal components: {X_pca.shape[1]}")
print(f"Components required for 95% variance: {components_95}")
print(
    f"Variance retained by selected components: "
    f"{cumulative_variance[components_95 - 1] * 100:.2f}%"
)

# =======================================================
# 6. SAVE EXPLAINED VARIANCE
# =======================================================

variance_df = pd.DataFrame({
    "principal_component": [
        f"PC{i + 1}" for i in range(len(explained_variance))
    ],
    "explained_variance_ratio": explained_variance,
    "explained_variance_percentage": explained_variance * 100,
    "cumulative_variance_percentage": cumulative_variance * 100,
})

variance_file = os.path.join(
    OUTPUT_DIR,
    "pca_explained_variance.csv"
)

variance_df.to_csv(variance_file, index=False)

print(f"\nExplained variance saved to:\n{variance_file}")

print("\nExplained variance by component:")
print(variance_df.round(4).to_string(index=False))

# =======================================================
# 7. SAVE REDUCED DATA
# =======================================================

reduced_df = pd.DataFrame(
    X_pca[:, :components_95],
    columns=[
        f"PC{i + 1}" for i in range(components_95)
    ]
)

reduced_file = os.path.join(
    OUTPUT_DIR,
    "pca_reduced_data.csv"
)

reduced_df.to_csv(reduced_file, index=False)

print(f"\nReduced data saved to:\n{reduced_file}")

# =======================================================
# 8. PLOT EXPLAINED VARIANCE
# =======================================================

plt.figure(figsize=(10, 6))

component_numbers = np.arange(
    1,
    len(cumulative_variance) + 1
)

plt.plot(
    component_numbers,
    cumulative_variance * 100,
    marker="o"
)

plt.axhline(
    y=95,
    linestyle="--",
    label="95% variance threshold"
)

plt.axvline(
    x=components_95,
    linestyle="--",
    label=f"{components_95} components"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("RideRush - PCA Cumulative Explained Variance")
plt.xticks(component_numbers)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

variance_plot = os.path.join(
    PLOTS_DIR,
    "pca_explained_variance.png"
)

plt.savefig(
    variance_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =======================================================
# 9. TWO-DIMENSIONAL PCA VISUALIZATION
# =======================================================

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=data["demand"],
    cmap="viridis",
    s=12,
    alpha=0.65
)

plt.xlabel(
    f"PC1 ({explained_variance[0] * 100:.2f}% variance)"
)

plt.ylabel(
    f"PC2 ({explained_variance[1] * 100:.2f}% variance)"
)

plt.title("RideRush - PCA Two-Dimensional Projection")

plt.colorbar(
    scatter,
    label="Taxi Demand"
)

plt.grid(True, alpha=0.2)
plt.tight_layout()

pca_plot = os.path.join(
    PLOTS_DIR,
    "pca_2d_visualization.png"
)

plt.savefig(
    pca_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =======================================================
# 10. FINAL SUMMARY
# =======================================================

print("\n" + "=" * 65)
print("PCA ANALYSIS COMPLETED")
print("=" * 65)

print(f"Original features: {len(available_features)}")
print(f"Components retaining 95% variance: {components_95}")

print("\nGenerated files:")
print("- output/pca_explained_variance.csv")
print("- output/pca_reduced_data.csv")
print("- plots/pca_explained_variance.png")
print("- plots/pca_2d_visualization.png")

print("\nNext step: evaluate the model pipeline and finalize RideRush.")