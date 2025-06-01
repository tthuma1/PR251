import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv('data/kvadrati_normalized.csv')

df['leto_gradnje'] = pd.to_numeric(df['leto_gradnje'], errors='coerce')
df['cena'] = df['cena'].str.split().str.get(0).str.replace('.', '', regex=False).str.replace(',', '.')
df['cena'] = pd.to_numeric(df['cena'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

df = df.dropna(subset=['latitude', 'longitude'])

st.title("Nepremičnine po Sloveniji")
st.markdown("### Zemljevid nepremičnin")

filter_year = st.checkbox("Leto gradnje")
if filter_year:
    min_year = int(df['leto_gradnje'].min(skipna=True))
    col1, col2 = st.columns(2)
    with col1:
        year_min_input = st.number_input("Min. leto gradnje", min_value=min_year, max_value=2025, value=min_year)
    with col2:
        year_max_input = st.number_input("Max. leto gradnje", min_value=min_year, max_value=2025, value=2025)
else:
    year_min_input = df['leto_gradnje'].min()
    year_max_input = df['leto_gradnje'].max()

filter_cena = st.checkbox("Cena")
if filter_cena:
    min_cena = int(df['cena'].min(skipna=True))
    max_cena = int(df['cena'].max(skipna=True))
    col3, col4 = st.columns(2)
    with col3:
        cena_min_input = st.number_input("Min. cena", min_value=0, max_value=max_cena, value=min_cena)
    with col4:
        cena_max_input = st.number_input("Max. cena", min_value=0, max_value=max_cena, value=max_cena)
else:
    cena_min_input = df['cena'].min()
    cena_max_input = df['cena'].max()

filtered_df = df[
    (df['leto_gradnje'] >= year_min_input) &
    (df['leto_gradnje'] <= year_max_input) &
    (df['cena'] >= cena_min_input) &
    (df['cena'] <= cena_max_input)
]

gdf = gpd.GeoDataFrame(
    filtered_df,
    geometry=[Point(xy) for xy in zip(filtered_df['longitude'], filtered_df['latitude'])],
    crs="EPSG:4326"
)

# Zemljevid Slovenije
fig, ax = plt.subplots(figsize=(9, 11), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([13.3, 16.6, 45.4, 47.2])  # cela Slovenija

ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
ax.add_feature(cfeature.LAND, facecolor='lightgrey')
ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)
ax.gridlines(draw_labels=True)

gdf.plot(ax=ax, color='red', markersize=5, label='Nepremičnine')

ax.set_title("Nepremičnine po Sloveniji")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

st.pyplot(fig)

st.subheader("Prikazane nepremičnine")
st.dataframe(filtered_df[['naslov', 'leto_gradnje', 'cena', 'latitude', 'longitude']])