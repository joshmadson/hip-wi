import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


df = pd.read_csv("G:\data\ibtracs\ibtracs.ALL.list.v04r01.csv",
                 header = 0,
                 skiprows = [2],
                 dtype = str
                 )


# Example Storm - Hurricane Milton


##
df["USA_SSHS"] = pd.to_numeric(df["USA_SSHS"], errors="coerce")
df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
df["DIST2LAND"]  = pd.to_numeric(df["DIST2LAND"], errors="coerce")

df_milton = df[
    (df["NAME"].str.upper() == "MILTON") &
    (df["USA_SSHS"] >= 1) &
    (df["SEASON"] == 2024)
].copy()

landfall_rows = df_milton
[
    (df_milton
["USA_SSHS"] >= 1) &
    (df_milton
["DIST2LAND"] == 0)
]


##
if not landfall_rows.empty:
    landfall_idx = landfall_rows.index[-1]  # last landfall index

    next_row = df[
        (df.index > landfall_idx) &
        (df["NAME"].str.upper() == "MILTON") &
        (df["SEASON"] == 2024) &
        (df["USA_SSHS"] == 0)
    ].head(1)

    df_milton_final = pd.concat([df_milton, next_row])
else:
    df_milton_final = df_milton

for col in ["USA_R64_NE", "USA_R64_SE", "USA_R64_SW", "USA_R64_NW"]:
    df_milton_final[col] = pd.to_numeric(df_milton_final[col], errors="coerce")

df_milton_final["buffer"] = df_milton_final[["USA_R64_NE", "USA_R64_SE", "USA_R64_SW", "USA_R64_NW"]].max(axis=1)

last_hurr_idx = df_milton_final[df_milton_final["USA_SSHS"] >= 1].index[-1]
last_hurr_buffer = df_milton_final.loc[last_hurr_idx, "buffer"]

final_trop_idx = df_milton_final[df_milton_final["USA_SSHS"] == 0].index[-1]

df_milton_final.loc[final_trop_idx, "buffer"] = 0.5 * last_hurr_buffer


##
df_milton_final["LAT"] = pd.to_numeric(df_milton_final["LAT"], errors="coerce")
df_milton_final["LON"] = pd.to_numeric(df_milton_final["LON"], errors="coerce")

gdf = gpd.GeoDataFrame(
    df_milton_final,
    geometry=gpd.points_from_xy(df_milton_final["LON"], df_milton_final["LAT"]),
    crs="EPSG:4326"   # WGS84 lat/lon
)

gdf = gdf.to_crs(epsg=3857)

gdf["geometry"] = gdf.buffer(gdf["buffer"] * 1609.34)
gdf = gdf.to_crs(epsg=4326)
gdf.plot(column="USA_SSHS", legend=True, figsize=(8,8))
plt.title("USDA-RMA Storm Buffers: Hurricane Milton")
plt.show()

##
import math
from shapely.geometry import LineString

def circle_tangents(x1, y1, r1, x2, y2, r2):
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)
    if d == 0:
        return []  # concentric circles, no tangents

    vx, vy = dx/d, dy/d

    # External tangents (change sign for internal tangents)
    result = []
    for sign in [+1, -1]:
        h = (r1 - r2) / d
        if abs(h) > 1:
            continue
        u = math.sqrt(max(0.0, 1 - h*h))
        ux = vx*h - sign*vy*u
        uy = vy*h + sign*vx*u

        # Tangent points
        x1t, y1t = x1 - r1*ux, y1 - r1*uy
        x2t, y2t = x2 - r2*ux, y2 - r2*uy

        result.append(LineString([(x1t, y1t), (x2t, y2t)]))

    return result


##
# --- Storm center points ---
gdf_points = gpd.GeoDataFrame(
    df_milton_final,
    geometry=gpd.points_from_xy(df_milton_final["LON"], df_milton_final["LAT"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

# --- Storm buffers (for plotting) ---
gdf_buffers = gdf_points.copy()
gdf_buffers["geometry"] = gdf_points.buffer(gdf_points["buffer"] * 1609.34)

# --- Filter to landfall points ---
landfall_points = gdf_points[gdf_points["DIST2LAND"] == 0].reset_index(drop=True)

# --- Compute tangents between consecutive landfall buffers ---
tangent_lines = []
for i in range(len(landfall_points) - 1):
    row1, row2 = landfall_points.iloc[i], landfall_points.iloc[i + 1]

    x1, y1 = row1.geometry.x, row1.geometry.y
    x2, y2 = row2.geometry.x, row2.geometry.y
    r1 = row1["buffer"] * 1609.34
    r2 = row2["buffer"] * 1609.34

    tangents = circle_tangents(x1, y1, r1, x2, y2, r2)
    tangent_lines.extend(tangents)

# --- Put tangents in a GeoDataFrame ---
gdf_tangents = gpd.GeoDataFrame(geometry=tangent_lines, crs="EPSG:3857")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 10))
gdf_buffers.plot(ax=ax, color="red", alpha=0.3, edgecolor="black")
gdf_tangents.plot(ax=ax, color="blue", linewidth=2)
landfall_points.plot(ax=ax, color="black", markersize=10)

plt.title("Tangent Lines Between Hurricane Buffers at Landfall")
plt.show()


##
import matplotlib.pyplot as plt
from shapely.ops import unary_union

# --- Step 1: Filter to landfall points ---
landfall_points = gdf_points[gdf_points["DIST2LAND"] == 0].reset_index(drop=True)

# Buffers only for landfall points
landfall_buffers = landfall_points.copy()
landfall_buffers["geometry"] = landfall_points.buffer(landfall_points["buffer"] * 1609.34)

# --- Step 2: Tangents between consecutive landfall buffers ---
tangent_lines = []
for i in range(len(landfall_points) - 1):
    row1, row2 = landfall_points.iloc[i], landfall_points.iloc[i+1]
    x1, y1, r1 = row1.geometry.x, row1.geometry.y, row1["buffer"] * 1609.34
    x2, y2, r2 = row2.geometry.x, row2.geometry.y, row2["buffer"] * 1609.34
    tangent_lines.extend(circle_tangents(x1, y1, r1, x2, y2, r2))

landfall_tangents = gpd.GeoDataFrame(geometry=tangent_lines, crs=landfall_points.crs)

# --- Step 3: Smoothed extent from landfall buffers + tangents ---
smoothed_extent = unary_union(list(landfall_buffers.geometry) + list(landfall_tangents.geometry)).buffer(0)
gdf_extent = gpd.GeoDataFrame(geometry=[smoothed_extent], crs=landfall_points.crs)

# --- Step 4: Intersect with counties in adjacent states ---
# Ensure CRS consistency
counties = counties.to_crs(landfall_points.crs)
states   = states.to_crs(landfall_points.crs)

states_touched = gpd.sjoin(states, gdf_extent, how="inner", predicate="intersects")
adjacent_states = states[states.intersects(states_touched.unary_union)]
counties_region = counties[counties["STATEFP"].isin(adjacent_states["STATEFP"])]

counties_touched = gpd.sjoin(counties_region, gdf_extent, how="inner", predicate="intersects")
counties_touched = counties_touched.drop_duplicates(subset="GEOID")

# --- Step 5: Plot ---
fig, ax = plt.subplots(figsize=(12,12))

# Base counties in region
counties_region.plot(ax=ax, color="white", edgecolor="lightgrey")

# Highlight intersecting counties
counties_touched.plot(ax=ax, color="lightblue", edgecolor="black")

# Landfall buffers
landfall_buffers.plot(ax=ax, color="red", alpha=0.2, edgecolor="black")

# Tangent lines
landfall_tangents.plot(ax=ax, color="blue", linewidth=1)

# Smoothed extent outline
gdf_extent.boundary.plot(ax=ax, color="green", linewidth=2)

# State boundaries
adjacent_states.boundary.plot(ax=ax, color="black", linewidth=1)

plt.title("Hurricane Milton: Smoothed Extent from Buffers and Intersecting Counties", fontsize=14)
plt.show()







## Plot unsmoothed extent with US Counties Hurricane Milton

import geopandas as gpd
import matplotlib.pyplot as plt

# Load US counties and states
counties = gpd.read_file(r"G:\data\maps\census_shapefiles\cb_2018_us_county_500k/cb_2018_us_county_500k.shp")
states = gpd.read_file(r"G:\data\maps\census_shapefiles\cb_2018_us_state_500k/cb_2018_us_state_500k.shp").to_crs(epsg=4326)

counties = counties.to_crs(epsg=4326)
states   = states.to_crs(epsg=4326)
gdf      = gdf.to_crs(epsg=4326)

# Spatial join: counties that intersect buffers
counties_touched = gpd.sjoin(counties, gdf, how="inner", predicate="intersects")
counties_touched = counties_touched.drop_duplicates(subset="GEOID")

# States that intersect storm circles
states_touched = gpd.sjoin(states, gdf, how="inner", predicate="intersects")
touched_state_fips = states_touched["STATEFP"].unique()

adjacent_states = states[states.intersects(states_touched.unary_union)]
adjacent_state_fips = adjacent_states["STATEFP"].unique()

counties_region = counties[counties["STATEFP"].isin(adjacent_state_fips)]

counties_touched = gpd.sjoin(counties_region, gdf, how="inner", predicate="intersects")
counties_touched = counties_touched.drop_duplicates(subset="GEOID")

fig, ax = plt.subplots(figsize=(10,10))

# All counties in region (black/white)
counties_region.plot(ax=ax, color="white", edgecolor="lightgrey")

# Touched counties (blue)
counties_touched.plot(ax=ax, color="lightblue", edgecolor="black")

# Buffers (red)
gdf.plot(ax=ax, color="red", alpha=0.3)

# State boundaries (black)
adjacent_states.boundary.plot(ax=ax, color="black", linewidth=1)

plt.title("Hurricane Milton: Storm Buffers and Effected Counties")
plt.show()