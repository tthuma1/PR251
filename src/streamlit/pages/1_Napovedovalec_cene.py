import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.applications.resnet50 import preprocess_input

import python_data

# --- Configuration ---
MODEL_PATH = "src/streamlit/data/image_model.keras"
IMAGE_SIZE = (224, 224)

# --- Load model and scaler ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- Image preprocessing ---
def preprocess_uploaded_image(file):
    image_bytes = file.read()  # 'file' is the file-like object
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = preprocess_input(img)
    return tf.expand_dims(img, axis=0)  # Add batch dimension

# --- Streamlit UI ---
st.title("Napovedovanje cene nepremičnine")

st.markdown(
    """
    Naš najboljši model za napoved cen je sestavljen iz dveh modelov. Prvi napove ceno iz slike in je implementiran s
    konvolucijsko nevronsko mrežo, naučeno na ImageNet-1k zbirki in prilagojen na našo zbirko slik nepremičnin. Drugi pa ceno napove iz opisnih
    atributov. Tu lahko izbiramo med različnimi modeli, ki vključujejo Gradient boosting, naključni gozd, Ridge, Lasso in linearno regresijo. Povprečna
    napaka modela je 28% cene.
    """
)

uploaded_file = st.file_uploader("Izberi sliko:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Naložena slika', use_container_width=True)

    with st.spinner("Razmišljam..."):
        try:
            preprocessed_img = preprocess_uploaded_image(uploaded_file)
            pred_scaled = model.predict(preprocessed_img)
            print(pred_scaled)
            image_prediction = np.expm1(pred_scaled).ravel()[0]  # Assuming single output
            # st.success(f"Napovedana cena: **{image_prediction:,.2f} €/m²**")
        except Exception as e:
            st.error(f"Napaka: {e}")


#####

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

nepremicnine = get_data('src/streamlit/data/nepremicnine_prodaja.csv')

nepremicnine_reg, nepremicnine_norm = get_formatted(nepremicnine)

col1, col2 = st.columns(2)

with col1:
    selected_type = st.selectbox(
        "Izberi tip nepremičnine:",
        get_unique(nepremicnine_reg["type"]),
    )

    selected_region = st.selectbox(
        "Izberi regijo:",
        get_unique(nepremicnine_reg["correct_region"]),
    )

with col2:
    selected_area = st.number_input(label="Vpiši površino:", min_value=1.0, step=0.01, value=80.0)

    selected_model = st.selectbox(
        "Izberi model:",
        models,
    )

pipeline = get_pipeline(model=selected_model, data=nepremicnine_reg)

new_data = pd.DataFrame({'correct_region': [selected_region], 'type': [selected_type], 'size': [selected_area]})
prediction = pipeline.predict(new_data)
attr_prediction = round(prediction[0], 2) / selected_area


if uploaded_file is None:
    st.success(f"Napovedana cena: **{attr_prediction:,.2f} €/m²**")
else:
    st.success(f"Napovedana cena: **{0.5*attr_prediction + 0.5*image_prediction:,.2f} €/m²**")