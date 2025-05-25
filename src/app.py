import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("real_estate_model.h5", compile=False)
scaler = joblib.load("price_scaler.pkl")

# Constants
IMAGE_SIZE = (224, 224)

st.title("🏠 Real Estate Price Estimator from Images")
st.write("Upload a photo of a property to estimate its price.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

    # Predict
    pred_scaled = model.predict(img_array)[0][0]
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]

    st.subheader("💰 Estimated Price:")
    st.write(f"**{pred_price:,.2f} currency units**")  # format as needed
