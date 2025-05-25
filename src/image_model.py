import os
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Paths
IMAGE_DIR = "../data/resized_images/"
CSV_PATH = "../data/kvadrati2/kvadrati_normalized.csv"
IMAGE_SIZE = (224, 224)

# Load CSV
df = pd.read_csv(CSV_PATH)
df['filename'] = df['id'].astype(str) + '.jpg'

def ocisti_ceno(cena_str, velikost):
    if pd.isna(cena_str):
        return np.nan
    cena_str = str(cena_str).lower().replace("€", "").replace("€", "").replace(".", "").strip()
    
    try:
        if "m2" in cena_str:
            vrednost_na_m2 = float(cena_str.replace("m2", "").strip())
            return vrednost_na_m2 * velikost
        else:
            return float(cena_str)
    except:
        return np.nan

df = df.dropna(subset=['cena']).reset_index(drop=True)
df['cena_clean'] = df.apply(lambda row: ocisti_ceno(row['cena'], row['velikost']), axis=1)

# Normalize prices if not already normalized
df = df.dropna(subset=['cena_clean']).reset_index(drop=True)
scaler = MinMaxScaler()
df['price_scaled'] = scaler.fit_transform(df[['cena_clean']])

# Load and preprocess images
def load_images_and_filter_df(df, image_dir, image_size=(224, 224)):
    images = []
    valid_indices = []

    for i, fname in enumerate(df['filename'].values):
        img_path = os.path.join(image_dir, fname)
        if i >= 1000: break
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(image_size)
                img_array = np.array(img) / 255.0
                images.append(img_array)
                valid_indices.append(i)
            except Exception as e:
                print(f"Skipping {fname}: {e}")
        else:
            print(f"Missing image: {fname}")

    # Return only rows from df that had valid images
    filtered_df = df.iloc[valid_indices].reset_index(drop=True)
    X = np.array(images)
    return X, filtered_df

print("Loading images...")

# Prepare dataset
image_dir = "../data/resized_images/"
X, filtered_df = load_images_and_filter_df(df, image_dir)
y = filtered_df['price_scaled'].values

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
print("Creating model...")
base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)  # Single output for price
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])

model.save("real_estate_model.keras", save_format="keras")

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=2,
    batch_size=32
)

# Plot training curves
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Model Training - Mean Absolute Error')
plt.show()

# Predict and inverse scale
pred_scaled = model.predict(X_val)
pred_prices = scaler.inverse_transform(pred_scaled)
true_prices = scaler.inverse_transform(y_val.reshape(-1, 1))

# Display a few results
for i in range(5):
    print(f"Predicted: {pred_prices[i][0]:,.2f}, Actual: {true_prices[i][0]:,.2f}")

import joblib
joblib.dump(scaler, "price_scaler.pkl")
