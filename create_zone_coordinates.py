import geopandas as gpd
import pandas as pd
from pathlib import Path

# ============================================================
# 1. FILE PATH
# ============================================================

shapefile = Path(
    r"C:\Users\boddu\PycharmProjects\RideRush\taxi_zones\taxi_zones.shp"
)

# ============================================================
# 2. CHECK FILE
# ============================================================

if not shapefile.exists():
    print("ERROR: Shapefile not found:")
    print(shapefile)
    exit()

print("Loading NYC taxi zones...")

# ============================================================
# 3. LOAD SHAPEFILE
# ============================================================

gdf = gpd.read_file(shapefile)

print("Columns found:")
print(gdf.columns.tolist())

# ============================================================
# 4. FIND LOCATION ID COLUMN
# ============================================================

if "LocationID" not in gdf.columns:

    print("ERROR: LocationID column not found.")
    print("Available columns:")
    print(gdf.columns.tolist())

    exit()

# ============================================================
# 5. PROJECT TO NYC METRIC CRS
# ============================================================

gdf = gdf.to_crs(epsg=2263)

# ============================================================
# 6. CALCULATE ZONE CENTROIDS
# ============================================================

gdf["centroid"] = gdf.geometry.centroid

# Convert centroid back to latitude/longitude
centroids = gdf.set_geometry("centroid").to_crs(epsg=4326)

# ============================================================
# 7. CREATE OUTPUT
# ============================================================

output = pd.DataFrame({

    "LocationID":
        centroids["LocationID"].astype(int),

    "latitude":
        centroids.geometry.y,

    "longitude":
        centroids.geometry.x
})

# ============================================================
# 8. SAVE
# ============================================================

output_path = Path(
    r"C:\Users\boddu\PycharmProjects\RideRush\zone_coordinates.csv"
)

output.to_csv(
    output_path,
    index=False
)

print()
print("SUCCESS!")
print("Created:")
print(output_path)

print()
print("Number of zones:", len(output))

print()
print(output.head())