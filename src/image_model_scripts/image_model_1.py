# made it to epoch 47
import os
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib

# --- TensorFlow GPU memory growth setup ---
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
AUTOTUNE = tf.data.AUTOTUNE

# --- Load and preprocess CSV ---
df = pd.read_csv(CSV_PATH)
df['filename'] = df['id'].astype(str) + '.jpg'

# --- Clean 'cena' column ---
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
df = df.dropna(subset=['cena_clean']).reset_index(drop=True)

# --- Normalize prices ---
scaler = MinMaxScaler()
df['price_scaled'] = scaler.fit_transform(df[['cena_clean']])

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

# --- Split dataset ---
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# --- TensorFlow data pipeline ---
def decode_image(filename, label):
    def _decode(filename_str, label_val):
        try:
            image_path = os.path.join(IMAGE_DIR, filename_str.numpy().decode())
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, IMAGE_SIZE)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label_val, True
        except Exception as e:
            print(f"⚠️ Skipping bad image: {filename_str.numpy().decode()} — {e}")
            dummy_image = np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)
            return dummy_image, label_val, False

    image, label, is_valid = tf.py_function(
        _decode, [filename, label], [tf.float32, tf.float32, tf.bool]
    )
    image.set_shape((*IMAGE_SIZE, 3))
    label.set_shape(())
    is_valid.set_shape(())

    return image, label, is_valid

def get_dataset(df, shuffle=True):
    filenames = df['filename'].values
    labels = df['price_scaled'].values.astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(decode_image, num_parallel_calls=AUTOTUNE)

    # Filter out invalid images
    dataset = dataset.filter(lambda img, lbl, valid: valid)
    
    # Drop the `valid` flag
    dataset = dataset.map(lambda img, lbl, _: (img, lbl), num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = get_dataset(train_df)
val_ds = get_dataset(val_df, shuffle=False)

# --- Build model ---
print("Creating model...")
base_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

# --- Save model architecture early ---
model.save("../models/real_estate_model.keras", save_format="keras")

# --- Train model ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100
)

# --- Plot training curves ---
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Model Training - Mean Absolute Error')
plt.savefig('model_train.png')
plt.show()

# --- Predict and inverse scale ---
pred_scaled = model.predict(val_ds)
true_scaled = np.concatenate([y for _, y in val_ds], axis=0)

pred_prices = scaler.inverse_transform(pred_scaled)
true_prices = scaler.inverse_transform(true_scaled.reshape(-1, 1))

# --- Show sample predictions ---
for i in range(5):
    print(f"Predicted: {pred_prices[i][0]:,.2f}, Actual: {true_prices[i][0]:,.2f}")

# --- Save scaler ---
joblib.dump(scaler, "../models/price_scaler.pkl")