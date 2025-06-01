# script v3
import os
from PIL import Image
import pandas as pd
import numpy as np
import keras_cv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanAbsolutePercentageError
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from transformers import ViTFeatureExtractor, TFViTModel


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

IMAGE_DIR = "../data/fixed_kvadrati_images/"
CSV_PATH = "../data/kvadrati2/kvadrati_normalized.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

# --- Load and preprocess CSV ---
df = pd.read_csv(CSV_PATH)
df = df[(df['vrsta'] == "Stanovanje") | (df['vrsta'] == "Hiša")]
df['filename'] = df['id'].astype(str) + '.jpg'

def ocisti_ceno(cena_str, velikost):
    if pd.isna(cena_str):
        return np.nan
    cena_str = str(cena_str).lower().replace("€", "").replace(".", "").strip()
    try:
        if "m2" in cena_str:
            vrednost_na_m2 = float(cena_str.replace("m2", "").strip())
            return vrednost_na_m2 * velikost
        else:
            return float(cena_str)
    except:
        return np.nan
    
def get_velikost(v):
    split_v = str(v).split()
    try:
        return float(split_v[0].replace(',', '.'))
    except ValueError:
        return None
    

df = df.dropna(subset=['cena']).reset_index(drop=True)
df['cena_clean'] = df.apply(lambda row: ocisti_ceno(row['cena'], row['velikost']), axis=1)
df = df[(df["cena_clean"].notnull()) & (df["cena_clean"] < 10_000_000)  & (df["cena_clean"] > 10_000)]
df = df.dropna(subset=['cena_clean']).reset_index(drop=True)

df['velikost_clean'] = df['velikost'].apply(get_velikost)
df['cena_na_m2'] = df['cena_clean'] / df['velikost_clean']
df = df[(df["cena_na_m2"] > 200) & (df["cena_na_m2"] < 7_000)]

# --- Filter out missing or invalid image files ---
def is_valid_image_file(fname):
    path = os.path.join(IMAGE_DIR, fname)
    if not os.path.exists(path):
        print(f"⚠️ Missing: {fname}")
        return False
    try:
        with Image.open(path) as img:
            img.verify()  # Validate image integrity
        return True
    except Exception as e:
        print(f"❌ Corrupt image: {fname} — {e}")
        return False

# nepremicnine dataset
# nep_csv_path = "../data/nepremicnine_prodaja_slike.csv"

# def price_from_name(name):
#     try:
#         trailer = name.split(',')[-1].strip()
#         m2 = float(trailer.replace("m2", "").replace(" ", "").replace(",", "."))
#         return m2
#     except:
#         return np.nan

# nep_df = pd.read_csv(nep_csv_path)
# nep_df = nep_df.dropna(subset=['price', 'id']).copy()

# # Add filename column
# nep_df['filename'] = nep_df['id'].astype(str) + '.jpg'
# nep_df = nep_df[(nep_df['price'] > 10_000) & (nep_df['price'] < 10_000_000)]

# nep_df.rename(columns={'price': 'cena_clean'}, inplace=True)
# nep_df['velikost_clean'] = nep_df['name'].apply(price_from_name)
# nep_df = nep_df.dropna(subset=['cena_clean', 'velikost_clean']).reset_index(drop=True)

# nep_df['cena_na_m2'] = nep_df['cena_clean'] / nep_df['velikost_clean']
# nep_df = nep_df[(nep_df["cena_na_m2"] > 200) & (nep_df["cena_na_m2"] < 7_000)]

# nep_df = nep_df[~nep_df['name'].str.contains('posest', case=False)].sort_values(by='name')


# # merge dataframes
# columns_to_keep = ['id', 'filename', 'cena_na_m2']
# df = pd.concat([df[columns_to_keep], nep_df[columns_to_keep]], ignore_index=True)

df['image_path'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))
df['price'] = np.log1p(df['cena_na_m2'])

print("Checking image file validity...")
df = df[df['filename'].apply(is_valid_image_file)].reset_index(drop=True)

print(df)
# print(nep_df)

## data frame is ready here

plt.hist(df["cena_na_m2"],bins='sqrt')
plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("Predicted vs Actual Prices")
plt.plot()
plt.grid()
plt.savefig('price_distribution.png')
plt.show()

# load test and val data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

def decode_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = preprocess_input(img)
    return img

def make_dataset(df):
    paths = df['image_path'].values
    prices = df['price'].values.astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, prices))
    
    def process(path, label):
        return decode_img(path), label
    
    dataset = dataset.map(process).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = make_dataset(train_df)
val_ds = make_dataset(val_df)



# === Vision Transformer Base ===
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
IMG_SIZE = feature_extractor.size['height']  # Usually 224

# === Load Pretrained ViT Model ===
vit_model = TFViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# === Build Full Model ===
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)

# ViT expects pixel values between 0 and 1, then normalized as below
def preprocess_vit(x):
    return (x - feature_extractor.image_mean) / feature_extractor.image_std

x = tf.keras.layers.Lambda(preprocess_vit)(inputs)
x = vit_model.vit(x)[0][:, 0, :]  # CLS token output
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
model.summary()

# === Train ===
checkpoint_cb = ModelCheckpoint(
    filepath="../models/model.keras",
    save_best_only=True,
    monitor="val_loss",
    mode="min"
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=[checkpoint_cb]
)




# loss, mae = model.evaluate(val_ds)
# print(f"Validation MAE: {mae:.2f}")

# val_mae = history.history['val_mae']

# plt.figure(figsize=(8, 5))
# plt.plot(val_mae, marker='o', label='Validation MAE')
# plt.title('Validation MAE over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("model_epochs.png")
# plt.show()

# # Predict log prices
# log_preds = model.predict(val_ds).flatten()
# # Get true log prices from val_ds
# true_log_prices = np.concatenate([y for x, y in val_ds], axis=0)
# # Convert back to original price scale
# preds = np.expm1(log_preds)
# true_prices = np.expm1(true_log_prices)
# # Compute absolute errors
# print("Some predictions:")
# print(preds[:5])
# print(true_prices[:5])
# abs_errors = np.abs(preds - true_prices)
# avg_mae = np.average(abs_errors)

# print("Absolute avg MAE:", avg_mae)

# plt.figure(figsize=(8, 6))
# plt.scatter(true_prices, abs_errors, alpha=0.5)
# plt.xlabel("True Price")
# plt.ylabel("Absolute Error")
# plt.title("Absolute Error vs True House Price")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("model_validation.png")
# plt.show()

# ## save best
# model.save("../models/model.keras")