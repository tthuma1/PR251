import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

# Constants
IMAGE_SIZE = (224, 224)
MODEL_PATH = "../models/v2/real_estate_model.keras"
SCALER_PATH = "../models/v2/price_scaler.pkl"

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

# Streamlit UI
st.title("🏠 Real Estate Price Estimator")
st.write("Upload an image of a property to estimate its price.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image_resized = image.resize(IMAGE_SIZE)
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    pred_scaled = model.predict(image_array)[0][0]
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]

    # Output
    st.subheader("💰 Estimated Price:")
    st.write(f"**{pred_price:,.2f} currency units**")
