import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv('src/streamlit/data/kvadrati_normalized.csv')

df['leto_gradnje'] = pd.to_numeric(df['leto_gradnje'], errors='coerce')
df['cena'] = df['cena'].str.split().str.get(0).str.replace('.', '', regex=False).str.replace(',', '.')
df['cena'] = pd.to_numeric(df['cena'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

df = df.dropna(subset=['latitude', 'longitude'])

def get_velikost(v):
    split_v = str(v).split()
    try:
        return float(split_v[0].replace(',', '.'))
    except ValueError:
        return None

df['velikost_clean'] = df['velikost'].apply(get_velikost)
df['cena_na_m2'] = df['cena'] / df['velikost_clean']

st.title("Nepremičnine po Sloveniji")
st.markdown(
    """
    ### Zemljevid nepremičnin

    Na spodnjem zemljevidu lahko raziskujete našo podatkovno zbirko z nastavljanjem filtrov o letu gradnje in ceni
    nepremičnin.
    """
)

filter_year = st.checkbox("Leto gradnje")
if filter_year:
    min_year = int(df['leto_gradnje'].min(skipna=True))
    col1, col2 = st.columns(2)
    with col1:
        year_min_input = st.number_input("Min. leto gradnje", min_value=min_year, max_value=2025, value=min_year, key="min_leto")
    with col2:
        year_max_input = st.number_input("Max. leto gradnje", min_value=min_year, max_value=2025, value=2025, key="max_leto")
else:
    year_min_input = df['leto_gradnje'].min()
    year_max_input = df['leto_gradnje'].max()

filter_cena = st.checkbox("Cena")
if filter_cena:
    min_cena = int(df['cena'].min(skipna=True))
    max_cena = int(df['cena'].max(skipna=True))
    col3, col4 = st.columns(2)
    with col3:
        cena_min_input = st.number_input("Min. cena", min_value=0, max_value=max_cena, value=min_cena, key="min_cena")
    with col4:
        cena_max_input = st.number_input("Max. cena", min_value=0, max_value=max_cena, value=max_cena, key="max_cena")
else:
    cena_min_input = df['cena'].min()
    cena_max_input = df['cena'].max()

with st.spinner("Pripravljam podatke..."):
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

with st.spinner("Rišem zemljevid..."):
    # Zemljevid Slovenije
    fig, ax = plt.subplots(figsize=(9, 11), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([13.3, 16.6, 45.4, 47.1])  # cela Slovenija

    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)
    ax.gridlines(draw_labels=True)

    gdf.plot(ax=ax, color='red', markersize=5, label='Nepremičnina')

    ax.set_title("Nepremičnine po Sloveniji")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

st.pyplot(fig)

st.subheader("Prikazane nepremičnine")
st.dataframe(filtered_df[['naslov', 'leto_gradnje', 'cena', 'latitude', 'longitude']])

########### agencije

st.markdown(
    """
    ### Vizualizacija ponudnikov

    Na spodnji vizualizaciji lahko opazujete ponudbe top ponudnikov nepremičnin glede na vpisane atribute. Najdite
    najboljšega prodajalca za vašo ciljno nepremičnino. V primeru, da več ponudnikov ustreza filtru, se izriše
    15 ponudnikov z največ oglasi. Prikazani so samo oglasi hiš in stanovanj.
    """
)

filter_year = st.checkbox("Leto gradnje", key=2)
if filter_year:
    min_year = int(df['leto_gradnje'].min(skipna=True))
    col1, col2 = st.columns(2)
    with col1:
        year_min_input = st.number_input("Min. leto gradnje", min_value=min_year, max_value=2025, value=min_year, key="pon_min_l")
    with col2:
        year_max_input = st.number_input("Max. leto gradnje", min_value=min_year, max_value=2025, value=2025, key="pon_max_l")
else:
    year_min_input = df['leto_gradnje'].min()
    year_max_input = df['leto_gradnje'].max()

filter_cena = st.checkbox("Cena", key=3)
if filter_cena:
    min_cena = int(df['cena'].min(skipna=True))
    max_cena = int(df['cena'].max(skipna=True))
    col3, col4 = st.columns(2)
    with col3:
        cena_min_input = st.number_input("Min. cena", min_value=0, max_value=max_cena, value=min_cena, key="pon_min_c")
    with col4:
        cena_max_input = st.number_input("Max. cena", min_value=0, max_value=max_cena, value=max_cena, key="pon_max_c")
else:
    cena_min_input = df['cena'].min()
    cena_max_input = df['cena'].max()

filter_vrsta = st.checkbox("Vrsta", key=4)
if filter_vrsta:
    selected_type = st.selectbox(
        "Izberi tip nepremičnine:",
        ['Hiša', 'Stanovanje'],
    )
else:
    selected_type = None

with st.spinner("Pripravljam podatke..."):
    if selected_type is not None:
        filtered_df2 = df[
            (df['leto_gradnje'] >= year_min_input) &
            (df['leto_gradnje'] <= year_max_input) &
            (df['cena'] >= cena_min_input) &
            (df['cena'] <= cena_max_input) &
            (df['vrsta'] == selected_type)
        ]
    else:
        filtered_df2 = df[
            (df['leto_gradnje'] >= year_min_input) &
            (df['leto_gradnje'] <= year_max_input) &
            (df['cena'] >= cena_min_input) &
            (df['cena'] <= cena_max_input)
        ]

import matplotlib.pyplot as plt
import seaborn as sns

filtered_df2 = filtered_df2[(filtered_df2['cena_na_m2'].notnull()) & (filtered_df2['cena_na_m2'] < 15000) & (filtered_df2['cena_na_m2'] > 50)]
# kvadrati_cene = df[(df['cena'].notnull())]
counts = filtered_df2['prodajalec_agencija'].value_counts()

# Izberi samo agencije z vsaj 45 vrsticami
valid_agencies = counts[counts >= (counts.iloc[14] if len(counts) >= 15 else 0)].sort_values().index
ordered_agencies = counts[counts >= (counts.iloc[14] if len(counts) >= 15 else 0)].sort_values(ascending=False).index

mean_values = filtered_df2.groupby('prodajalec_agencija')['cena_na_m2'].mean().loc[ordered_agencies]

# Filtriraj DataFrame, da vsebuje le validne agencije
filtered_df = filtered_df2[(filtered_df2['prodajalec_agencija'].isin(valid_agencies))]
filtered_df = filtered_df[(filtered_df["vrsta"] == "Stanovanje") | (filtered_df["vrsta"] == "Hiša")]

filtered_df = filtered_df[filtered_df['id'].isin(filtered_df2['id'])]

# Ustvari scatter plot za filtrirane in sortirane agencije
fig = plt.figure(figsize=(12, 6))

sns.stripplot(
    data=filtered_df,
    y='prodajalec_agencija',
    x='cena_na_m2',
    order=ordered_agencies,
    alpha=0.7,
    hue='prodajalec_agencija',
    palette='Set2', 
    legend=None
)
sns.scatterplot(
    y=ordered_agencies,
    x=mean_values.values,
    color='r',
    marker='o',
    s=50,
    zorder=10,
    label="Povprečje"
)

plt.title('Cene na m² za prodajalce', fontsize=14)
plt.xlabel('Agencija', fontsize=12)
plt.ylabel('Cena na m² (€)', fontsize=12)
plt.tight_layout()

st.pyplot(fig)

st.subheader("Prikazane nepremičnine")
st.dataframe(filtered_df[['naslov', 'leto_gradnje', 'cena', "vrsta", 'prodajalec_agencija', 'latitude', 'longitude']])
