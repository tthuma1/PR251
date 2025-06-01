import os
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# --- GPU memory growth ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- Paths ---
IMAGE_DIR = "../data/kvadrati_images/"
CSV_PATH = "../data/kvadrati2/kvadrati_normalized.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 7

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

df = df.dropna(subset=['cena']).reset_index(drop=True)
df['cena_clean'] = df.apply(lambda row: ocisti_ceno(row['cena'], row['velikost']), axis=1)
df = df[(df["cena_clean"].notnull()) & (df["cena_clean"] < 1_000_000)  & (df["cena_clean"] > 10_000)]
df = df.dropna(subset=['cena_clean']).reset_index(drop=True)

# --- Class assignment using quantiles ---
df['luxury_class'] = pd.qcut(df['cena_clean'], q=NUM_CLASSES, labels=False)

# --- Filter invalid image files ---
def is_valid_image_file(fname):
    path = os.path.join(IMAGE_DIR, fname)
    if not os.path.exists(path):
        print(f"⚠️ Missing: {fname}")
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"❌ Corrupt image: {fname} — {e}")
        return False

df = df[df['filename'].apply(is_valid_image_file)].reset_index(drop=True)

# --- Train/val split ---
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['luxury_class'], random_state=42)

# --- TF Dataset pipeline ---
AUTOTUNE = tf.data.AUTOTUNE

def decode_image(filename, label):
    def _decode(filename_str, label_val):
        image_path = os.path.join(IMAGE_DIR, filename_str.numpy().decode())
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, IMAGE_SIZE)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label_val, True
        except Exception as e:
            print(f"⚠️ Skipping: {filename_str.numpy().decode()} — {e}")
            dummy_image = np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)
            return dummy_image, label_val, False

    image, label, valid = tf.py_function(
        _decode, [filename, label], [tf.float32, tf.int64, tf.bool]
    )
    image.set_shape((*IMAGE_SIZE, 3))
    label.set_shape(())
    valid.set_shape(())
    return image, label, valid

def get_dataset(df, shuffle=True):
    filenames = df['filename'].values
    labels = df['luxury_class'].values
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(decode_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.filter(lambda img, lbl, valid: valid)
    dataset = dataset.map(lambda img, lbl, _: (img, lbl), num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1000)
    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = get_dataset(train_df)
val_ds = get_dataset(val_df, shuffle=False)

# --- Model definition ---
base_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Train model ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# --- Plot training history ---
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.savefig('luxury_class_training.png')
plt.show()

# --- Save model and class encoder ---
model.save("../models/luxury_class_model.keras")
joblib.dump({'class_map': df[['cena_clean', 'luxury_class']]}, "../models/luxury_class_meta.pkl")
