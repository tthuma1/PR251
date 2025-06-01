import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import joblib

# --- Configuration ---
MODEL_PATH = "src/streamlit/data/image_model.keras"
IMAGE_SIZE = (224, 224)

# --- Load model and scaler ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- Image preprocessing ---
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dim
    return image_array

# --- Streamlit UI ---
st.title("Napovedovanje cene iz slike nepremičnine")
st.markdown(
    """
    Na tej strani je 
    """
)

uploaded_file = st.file_uploader("Izberi sliko", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Naložena slika:', use_container_width=True)

    with st.spinner("Razmišljam..."):
        try:
            img_array = preprocess_image(uploaded_file)
            pred_scaled = model.predict(img_array).flatten()[0]
            print(pred_scaled)
            pred_price = np.expm1(pred_scaled);
            st.success(f"Napovedana cena: **{pred_price:,.2f} €/m2**")
        except Exception as e:
            st.error(f"Napaka: {e}")


