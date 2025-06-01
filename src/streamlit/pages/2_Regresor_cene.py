import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import python_data


@st.cache_data
def get_data(url):
    res = pd.read_csv(url, sep=",")
    return res

@st.cache_data
def get_formatted(nepremicnine):
    nepremicnine_reg = nepremicnine[(nepremicnine["municipality"].isin(python_data.municipalities))].copy()

    nepremicnine_reg['price'] = pd.to_numeric(nepremicnine_reg['price'], errors='coerce')
    nepremicnine_reg.dropna(subset=['price'], inplace=True)

    nepremicnine_reg["correct_region"] = nepremicnine_reg["municipality"].apply(lambda x: python_data.municipalities[x])

    def size_from_name(name):
        try:
            trailer = name.split(',')[-1].strip()
            m2 = float(trailer.replace("m2", "").replace(" ", "").replace(",", "."))
            return m2
        except:
            return np.nan

    nepremicnine_reg['size'] = nepremicnine_reg['name'].apply(size_from_name)

    nepremicnine_reg = nepremicnine_reg[(nepremicnine_reg["price"] >= 10000) & (nepremicnine_reg["price"] < 100000000)][
        ["type", "correct_region", "size", "price"]]
    nepremicnine_reg.dropna(subset=['size'], inplace=True)

    nepremicnine_reg = nepremicnine_reg[(nepremicnine_reg['price'] / nepremicnine_reg['size']) < 25000]

    nepremicnine_norm = nepremicnine_reg.copy()
    nepremicnine_norm["price_per_m2"] = nepremicnine_norm["price"] / nepremicnine_norm["size"]
    nepremicnine_norm = nepremicnine_norm.drop('price', axis=1)
    nepremicnine_norm = nepremicnine_norm.drop('size', axis=1)

    return nepremicnine_reg, nepremicnine_norm

@st.cache_data
def get_unique(data):
    res = data.unique()
    return res

@st.cache_data
def get_preprocessor():
    categorical_features = ['correct_region', 'type']
    numerical_features = ['size']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ],
        remainder='passthrough',
        force_int_remainder_cols=False
    )
    return preprocessor

@st.cache_resource
def get_pipeline(model, data):
    X = data[["type", "correct_region", "size"]]
    y = data["price"]

    categorical_features = ['correct_region', 'type']
    numerical_features = ['size']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ],
        remainder='passthrough',
        force_int_remainder_cols=False
    )

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('regresija', models[model])
    ])

    pipeline.fit(X, y)

    return pipeline



models = {
    'GradientBoosting': GradientBoostingRegressor(),
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'RandomForest_depth5': RandomForestRegressor(max_depth=5),
    'RandomForest_depth6': RandomForestRegressor(max_depth=6),
}


# nepremicnine = pd.read_csv('../data/nepremicnine_prodaja.csv', sep=",")

# Getting the data set up for the regressor
#region Data set-up

# nepremicnine_reg = nepremicnine[(nepremicnine["municipality"].isin(python_data.municipalities))].copy()
#
# nepremicnine_reg['price'] = pd.to_numeric(nepremicnine_reg['price'], errors='coerce')
# nepremicnine_reg.dropna(subset=['price'], inplace=True)
#
# nepremicnine_reg["correct_region"] = nepremicnine_reg["municipality"].apply(lambda x: python_data.municipalities[x])
#
#
# def size_from_name(name):
#     try:
#         trailer = name.split(',')[-1].strip()
#         m2 = float(trailer.replace("m2", "").replace(" ", "").replace(",", "."))
#         return m2
#     except:
#         return np.nan
#
# nepremicnine_reg['size'] = nepremicnine_reg['name'].apply(size_from_name)
#
# nepremicnine_reg = nepremicnine_reg[(nepremicnine_reg["price"] >= 10000) & (nepremicnine_reg["price"] < 100000000)][["type", "correct_region", "size", "price"]]
# nepremicnine_reg.dropna(subset=['size'], inplace=True)
#
# nepremicnine_reg = nepremicnine_reg[(nepremicnine_reg['price'] / nepremicnine_reg['size']) < 25000]
#
# nepremicnine_norm = nepremicnine_reg.copy()
# nepremicnine_norm["price_per_m2"] = nepremicnine_norm["price"] / nepremicnine_norm["size"]
# nepremicnine_norm = nepremicnine_norm.drop('price', axis=1)
# nepremicnine_norm = nepremicnine_norm.drop('size', axis=1)

#endregion Data set-up

nepremicnine = get_data('../../data/nepremicnine/nepremicnine_prodaja.csv')

nepremicnine_reg, nepremicnine_norm = get_formatted(nepremicnine)

st.title('Napovedovanje cene glede na atribute')

selected_type = st.selectbox(
    "Izberite tip nepremičnine:",
    get_unique(nepremicnine_reg["type"]),
)

selected_region = st.selectbox(
    "Izberite regijo:",
    get_unique(nepremicnine_reg["correct_region"]),
)

selected_area = st.number_input(label="Vpišite površino:", min_value=0.0, step=0.01)

selected_model = st.selectbox(
    "Izberite model:",
    models,
)

pipeline = get_pipeline(model=selected_model, data=nepremicnine_reg)

new_data = pd.DataFrame({'correct_region': [selected_region], 'type': [selected_type], 'size': [selected_area]})
prediction = pipeline.predict(new_data)

st.write(f"Napovedana cena: **{str(round(prediction[0], 2))}** EUR")
