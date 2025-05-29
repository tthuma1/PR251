import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import joblib
import os

# --- Constants ---
IMAGE_SIZE = (224, 224)
MODEL_PATH = "../models/v9/multi_input_model.keras"
PREPROCESSOR_PATH = "../models/v9/preprocessor.pkl"
SCALER_PATH = "../models/v9/price_scaler.pkl"

# --- Load model and preprocessing tools ---
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# --- UI ---
st.title("🏠 Real Estate Price Predictor")
st.markdown("Upload a photo and fill in the property details (optional).")


uploaded_file = st.file_uploader("Upload an image of the property", type=["jpg", "jpeg", "png"])


# --- Example structured inputs ---
col1, col2 = st.columns(2)
with col1:
    velikost = st.number_input("Size (m²)", value=None)
    leto_gradnje = st.number_input("Year Built", min_value=1800, max_value=2050, value=None)
    tip = st.selectbox("Type", options=["", "1-bedroom", "2-bedroom", "House", "Studio"])
    prodajalec = st.selectbox("Seller Type", options=["", "Agency", "Owner"])
with col2:
    ener_class = st.selectbox("Energy Class", options=["", "A+", "A", "B", "C", "D", "E", "F", "G"])
    ogrevanje = st.multiselect("Heating", options=[
        "Biomasa", "CK", "Drva", "Elektrika", "Fotovoltaika", "Kamin", "Klima",
        "Olje", "Plin", "Rekuperator", "Sončni kolektorji", "Talno gretje", 
        "Toplarna", "Toplotna črpalka"
    ])
    dodatki = st.multiselect("Extras", options=[
        "Balkon", "Terasa", "Klet", "Dvorišče", "Vrt", "Garaža", "Loža", "Atrij"
    ])

# --- Predict Button ---
if st.button("Predict Price"):

    # --- Validate image ---
    if not uploaded_file:
        st.warning("Please upload an image.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0
    except Exception as e:
        st.error(f"Image loading failed: {e}")
        st.stop()

    # --- Prepare structured input ---
    input_dict = {
        "velikost_clean": velikost,
        "leto_gradnje": leto_gradnje,
        "tip": tip,
        "prodajalec_agencija": 1 if prodajalec == "Agency" else 0,
        "Ener. izk.": ener_class,
    }

    # Add multiselect categorical features
    for feat in ["Balkon", "Terasa", "Klet", "Dvorišče", "Vrt", "Garaža", "Loža", "Atrij"]:
        input_dict[f"Dodatno_{feat}"] = 1 if feat in dodatki else 0

    for feat in ogrevanje:
        input_dict[f"Ogrevanje_{feat}"] = 1

    # Ensure input includes all expected features from training
    expected_cols = list(preprocessor.feature_names_in_)
    input_data = {}

    for col in expected_cols:
        # Fill known values
        if col in input_dict:
            input_data[col] = input_dict[col]
        # Fill all extras, binary and categorical, with 0 or empty string
        elif col.startswith("Dodatno_") or col.startswith("Ogrevanje_") or col.startswith("Lega_") or col.startswith("Luksuz_") or col.startswith("Priključki_"):
            input_data[col] = 0
        elif col in ["tip", "Ener. izk."]:
            input_data[col] = ""
        else:
            input_data[col] = 0.0

    # Convert to DataFrame for transformer
    import pandas as pd
    input_df = pd.DataFrame([input_data])
    structured_input = preprocessor.transform(input_df).astype(np.float32)

    st.write("Structured input:", structured_input)
    st.image(uploaded_file, caption="Input image", use_column_width=True)

    # --- Prepare image and predict ---
    image_tensor = tf.convert_to_tensor([image_array], dtype=tf.float32)

    pred_scaled = model.predict([image_tensor, structured_input])[0][0]
    predicted_price = scaler.inverse_transform([[pred_scaled]])[0][0]

    st.success(f"💰 Estimated Price: **{predicted_price:,.2f} €**")
