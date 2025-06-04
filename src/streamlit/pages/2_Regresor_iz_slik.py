import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import joblib
from tensorflow.keras.applications.resnet50 import preprocess_input

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
st.title("Napovedovanje cene iz slike nepremičnine")
st.markdown(
    """
    Na tej strani je napoved cene izvedena izključno z modelom za napoved iz slike, ki je opisan na strani
    [Napovedovalec cene](Napovedovalec_cene). Povprečna napaka modela je 33%.
    """
)

uploaded_file = st.file_uploader("Izberi sliko", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Naložena slika:', use_container_width=True)

    with st.spinner("Razmišljam..."):
        try:
            preprocessed_img = preprocess_uploaded_image(uploaded_file)
            pred_scaled = model.predict(preprocessed_img)
            print(pred_scaled)
            pred_price = np.expm1(pred_scaled).ravel()[0]  # Assuming single output
            st.success(f"Napovedana cena: **{pred_price:,.2f} €/m²**")
        except Exception as e:
            st.error(f"Napaka: {e}")


