import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import ResNet50
import joblib

# Paths
IMAGE_DIR = "../data/kvadrati_images/"
CSV_PATH = "../data/kvadrati2/kvadrati_normalized.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Load and preprocess CSV
df = pd.read_csv(CSV_PATH)
df['filename'] = df['id'].astype(str) + '.jpg'

# Clean 'cena' column
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
df['velikost_clean'] = df['velikost'].apply(get_velikost)
df['cena_clean'] = df.apply(lambda row: ocisti_ceno(row['cena'], row['velikost']), axis=1)
df = df.dropna(subset=['cena_clean']).reset_index(drop=True)

# Filter out invalid images
def is_valid_image_file(fname):
    path = os.path.join(IMAGE_DIR, fname)
    if not os.path.exists(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

df = df[df['filename'].apply(is_valid_image_file)].reset_index(drop=True)

# Structured features
numerical = ['velikost_clean', 'leto_gradnje']
categorical = ['Ener. izk.', 'prodajalec_agencija', 'tip']
binary = [col for col in df.columns if col.startswith(('Dodatno_', 'Lega_', 'Luksuz_', 'Ogrevanje_', 'Priključki_'))]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('bin', 'passthrough', binary)
])

X_structured = preprocessor.fit_transform(df[numerical + categorical + binary])
joblib.dump(preprocessor, "../models/preprocessor.pkl")

# Image preprocessing
def load_image(filename):
    path = os.path.join(IMAGE_DIR, filename)
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image / 255.0

images = np.stack([load_image(fname).numpy() for fname in df['filename']])

# Targets
y = df['cena_clean'].values

# Train-test split
X_img_train, X_img_val, X_struct_train, X_struct_val, y_train, y_val = train_test_split(
    images, X_structured, y, test_size=0.2, random_state=42
)

# Build model
image_input = Input(shape=(*IMAGE_SIZE, 3), name="image_input")
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
base_model.trainable = False
x1 = layers.GlobalAveragePooling2D()(base_model.output)

structured_input = Input(shape=(X_structured.shape[1],), name="structured_input")
x2 = layers.Dense(128, activation='relu')(structured_input)
x2 = layers.Dropout(0.3)(x2)

combined = layers.Concatenate()([x1, x2])
x = layers.Dense(128, activation='relu')(combined)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1)(x)

model = models.Model(inputs=[image_input, structured_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()

# Train model
history = model.fit(
    [X_img_train, X_struct_train], y_train,
    validation_data=([X_img_val, X_struct_val], y_val),
    epochs=50,
    batch_size=BATCH_SIZE
)

# Save model
model.save("../models/image_structured_model.keras")
