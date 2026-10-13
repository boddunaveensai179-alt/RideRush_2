"""
RideRush - ETA Model Comparison

Compares saved ETA evaluation results using:
- MAE
- RMSE
- R2

Important:
Models trained using different feature definitions or test sets
are not necessarily directly comparable.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = BASE_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. RESULT FILES
# ============================================================

RESULT_FILES = {
    "Decision Tree": "decision_tree_results.csv",
    "Gradient Boosting": "gradient_boosting_results.csv",
    "XGBoost": "xgboost_results.csv",
    "LightGBM": "lightgbm_results.csv",
    "Random Forest": "eta_rf_updated_metrics.csv",
}

results = []


# ============================================================
# 3. LOAD SAVED RESULTS
# ============================================================

print("=" * 65)
print("RIDErush - ETA MODEL COMPARISON")
print("=" * 65)

for model_name, filename in RESULT_FILES.items():

    path = OUTPUT_DIR / filename

    if not path.exists():
        print(f"Skipping missing file: {filename}")
        continue

    try:
        df = pd.read_csv(path)
    except Exception as error:
        print(f"Could not read {filename}: {error}")
        continue

    if df.empty:
        print(f"Skipping empty file: {filename}")
        continue

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    required = {"mae", "rmse", "r2"}

    if not required.issubset(df.columns):
        print(
            f"Skipping {filename}: "
            "MAE, RMSE, or R2 is missing."
        )
        continue

    for _, row in df.iterrows():

        try:
            mae = float(row["mae"])
            rmse = float(row["rmse"])
            r2 = float(row["r2"])

            if not all(
                pd.notna(value)
                for value in [mae, rmse, r2]
            ):
                continue

            result = {
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Source": filename,
            }

            # Preserve metadata when available.
            if "rows" in df.columns:
                result["Rows"] = row["rows"]

            if "distance_definition" in df.columns:
                result["Distance Definition"] = (
                    row["distance_definition"]
                )

            results.append(result)

        except (TypeError, ValueError):
            print(
                f"Skipping invalid metrics in {filename}"
            )


# ============================================================
# 4. VALIDATE RESULTS
# ============================================================

if not results:
    raise RuntimeError(
        "No valid ETA evaluation results were found. "
        "Run the training scripts first."
    )

comparison = pd.DataFrame(results)

# Sort by MAE: lower is better.
comparison = comparison.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

print("\n========== ETA MODEL COMPARISON ==========")

display_columns = [
    column
    for column in [
        "Model",
        "MAE",
        "RMSE",
        "R2",
        "Rows",
        "Distance Definition",
        "Source",
    ]
    if column in comparison.columns
]

print(
    comparison[display_columns]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 6. SAVE COMPARISON
# ============================================================

comparison_path = (
    OUTPUT_DIR / "model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print(f"\nComparison saved:\n{comparison_path}")


# ============================================================
# 7. PLOT MAE
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    comparison["Model"],
    comparison["MAE"]
)

plt.xlabel("ETA Model")
plt.ylabel("Mean Absolute Error (minutes)")
plt.title("RideRush - ETA Model Comparison by MAE")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

mae_plot_path = (
    PLOTS_DIR / "eta_model_comparison_mae.png"
)

plt.savefig(
    mae_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. PLOT RMSE
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    comparison["Model"],
    comparison["RMSE"]
)

plt.xlabel("ETA Model")
plt.ylabel("Root Mean Squared Error (minutes)")
plt.title("RideRush - ETA Model Comparison by RMSE")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

rmse_plot_path = (
    PLOTS_DIR / "eta_model_comparison_rmse.png"
)

plt.savefig(
    rmse_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

best_model = comparison.iloc[0]

print("\n========== PRELIMINARY BEST MODEL ==========")

print(f"Model: {best_model['Model']}")
print(f"MAE: {best_model['MAE']:.4f} minutes")
print(f"RMSE: {best_model['RMSE']:.4f} minutes")
print(f"R2: {best_model['R2']:.4f}")

print(
    "\nNote: The ranking is preliminary if models were "
    "evaluated using different test sets or feature definitions."
)

print("\nGenerated files:")
print("- output/model_comparison.csv")
print("- plots/eta_model_comparison_mae.png")
print("- plots/eta_model_comparison_rmse.png")

print("\nETA model comparison completed.")