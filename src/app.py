import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import joblib

# --- Configuration ---
MODEL_PATH = "../models/v11/real_estate_model.keras"
SCALER_PATH = "../models/v11/price_scaler.pkl"
IMAGE_SIZE = (224, 224)

# --- Load model and scaler ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_model()
scaler = load_scaler()

# --- Image preprocessing ---
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dim
    return image_array

# --- Streamlit UI ---
st.title("🏠 Real Estate Price Predictor")
st.write("Upload an image of the real estate property to predict its price.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Predicting..."):
        try:
            img_array = preprocess_image(uploaded_file)
            pred_scaled = model.predict(img_array)
            print(pred_scaled)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            st.success(f"💰 Estimated Price: **{pred_price:,.2f} €**")
        except Exception as e:
            st.error(f"Error: {e}")
