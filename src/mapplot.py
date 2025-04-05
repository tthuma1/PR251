import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

df = pd.read_csv('kvadrati.csv')
df = df[df[['longitude', 'latitude']].notnull().all(1)]

# Create Point geometries
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # WGS84

# Load world map and filter for Slovenia
world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")

# Filter Slovenia
slovenia = world[world['ADMIN'] == 'Slovenia']

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
slovenia.plot(ax=ax, color='lightgray')
gdf.plot(ax=ax, color='red', markersize=20)
plt.title("Points in Slovenia")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()



# import folium

# # Center the map around Slovenia
# m = folium.Map(location=[46.1512, 14.9955], zoom_start=8)

# # Add points
# for _, row in df.iterrows():
#     folium.Marker(location=[row['latitude'], row['longitude']]).add_to(m)

# # Save or display
# m.save("slovenia_map.html")