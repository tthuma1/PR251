# script v3
import os
from PIL import Image
import pandas as pd
import numpy as np
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
df = df[(df["cena_clean"].notnull()) & (df["cena_clean"] < 1_000_000)  & (df["cena_clean"] > 10_000)]
df = df.dropna(subset=['cena_clean']).reset_index(drop=True)

df['velikost_clean'] = df['velikost'].apply(get_velikost)
df['cena_na_m2'] = df['cena_clean'] / df['velikost_clean']
df = df[(df["cena_na_m2"] > 200) & (df["cena_na_m2"] < 7_000)]

df['image_path'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))

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

print("Checking image file validity...")
df = df[df['filename'].apply(is_valid_image_file)].reset_index(drop=True)

# df['price'] = np.log1p(df['cena_na_m2'])
print(df)

avg = df['cena_na_m2'].mean()
mae = (df['cena_na_m2'] - avg).abs().mean()
rmse = np.sqrt(((df['cena_na_m2'] - avg) ** 2).mean())
mape = ((df['cena_na_m2'] - avg).abs() / df['cena_na_m2']).mean() * 100

print("Avg. price:", avg)
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

## data frame is ready here