import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("plots", exist_ok=True)

df = pd.read_parquet(
    "output/riderush_cleaned_data.parquet"
)

print("=" * 55)
print("RIDErush - EDA")
print("=" * 55)

# 1. Trip duration
plt.figure(figsize=(8, 5))
sns.histplot(df["trip_duration"], bins=50, kde=True)
plt.title("Trip Duration Distribution")
plt.xlabel("Duration (minutes)")
plt.ylabel("Trips")
plt.tight_layout()
plt.savefig("plots/01_duration.png")
plt.show()

# 2. Trip distance
plt.figure(figsize=(8, 5))
sns.histplot(df["trip_distance"], bins=50, kde=True)
plt.title("Trip Distance Distribution")
plt.xlabel("Distance (miles)")
plt.ylabel("Trips")
plt.tight_layout()
plt.savefig("plots/02_distance.png")
plt.show()

# 3. Demand by hour
hourly = df.groupby("pickup_hour").size()

plt.figure(figsize=(9, 5))
plt.bar(hourly.index, hourly.values)
plt.title("Taxi Demand by Hour")
plt.xlabel("Hour")
plt.ylabel("Trips")
plt.xticks(range(24))
plt.tight_layout()
plt.savefig("plots/03_hourly_demand.png")
plt.show()

# 4. Demand by day
daily = df.groupby("pickup_dayofweek").size()

days = [
    "Mon", "Tue", "Wed",
    "Thu", "Fri", "Sat", "Sun"
]

plt.figure(figsize=(8, 5))
plt.bar(days, daily.reindex(range(7)))
plt.title("Taxi Demand by Day")
plt.xlabel("Day")
plt.ylabel("Trips")
plt.tight_layout()
plt.savefig("plots/04_daily_demand.png")
plt.show()

# 5. Distance vs duration
sample = df.sample(
    min(30000, len(df)),
    random_state=42
)

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=sample,
    x="trip_distance",
    y="trip_duration",
    alpha=0.3
)
plt.title("Distance vs Trip Duration")
plt.xlabel("Distance (miles)")
plt.ylabel("Duration (minutes)")
plt.tight_layout()
plt.savefig("plots/05_distance_duration.png")
plt.show()

# 6. Rush hour vs duration
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=sample,
    x="is_rush_hour",
    y="trip_duration"
)
plt.title("Rush Hour vs Trip Duration")
plt.xlabel("Rush Hour")
plt.ylabel("Duration (minutes)")
plt.tight_layout()
plt.savefig("plots/06_rush_duration.png")
plt.show()

# 7. Correlation
cols = [
    "trip_distance",
    "passenger_count",
    "fare_amount",
    "trip_duration"
]

corr = df[cols].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/07_correlation.png")
plt.show()

# 8. Top pickup locations
top_pickup = (
    df["PULocationID"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(9, 5))
plt.bar(
    top_pickup.index.astype(str),
    top_pickup.values
)
plt.title("Top Pickup Locations")
plt.xlabel("Location ID")
plt.ylabel("Trips")
plt.tight_layout()
plt.savefig("plots/08_top_pickup.png")
plt.show()

# 9. Demand heatmap
heatmap = (
    df.groupby([
        "pickup_dayofweek",
        "pickup_hour"
    ])
    .size()
    .unstack(fill_value=0)
)

plt.figure(figsize=(10, 5))
sns.heatmap(heatmap, cmap="YlOrRd")
plt.title("Demand by Day and Hour")
plt.xlabel("Hour")
plt.ylabel("Day")
plt.tight_layout()
plt.savefig("plots/09_demand_heatmap.png")
plt.show()

# 10. Average duration by hour
avg_duration = (
    df.groupby("pickup_hour")["trip_duration"]
    .mean()
)

plt.figure(figsize=(9, 5))
plt.plot(
    avg_duration.index,
    avg_duration.values,
    marker="o"
)
plt.title("Average Trip Duration by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Duration (minutes)")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/10_average_duration.png")
plt.show()

print("\nEDA completed.")
print("10 important EDA graphs saved in /plots")