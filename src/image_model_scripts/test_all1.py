# TODO: from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import os
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
CSV_PATH = "../data/kvadrati2/kvadrati_normalized.csv"
IMAGE_DIR = "../data/kvadrati_images/"
MODEL_PATH = "../models/v10/real_estate_model.keras"
SCALER_PATH = "../models/v10/price_scaler.pkl"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

# --- Load and preprocess CSV ---
df = pd.read_csv(CSV_PATH)
df = df[(df['vrsta'] == "Stanovanje") | (df['vrsta'] == "Hiša")]
df['filename'] = df['id'].astype(str) + '.jpg'

df['image_path'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))

# nepremicnine dataset
new_csv_path = "../data/nepremicnine_prodaja_slike.csv"

new_df = pd.read_csv(new_csv_path)
new_df = new_df.dropna(subset=['price', 'id']).copy()

# Add filename column
new_df['filename'] = new_df['id'].astype(str) + '.jpg'
new_df = new_df[(new_df['price'] > 10_000) & (new_df['price'] < 1_000_000)]

new_df.rename(columns={'price': 'cena_clean'}, inplace=True)


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
df = df[(df["cena_clean"].notnull()) & (df["cena_clean"] < 1_000_000)  & (df["cena_clean"] > 10_000)]
df = df.dropna(subset=['cena_clean']).reset_index(drop=True)

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
# new_df = new_df[new_df['filename'].apply(is_valid_image_file)].reset_index(drop=True)

# columns_to_keep = ['id', 'filename', 'cena_clean']
# merged_df = pd.concat([df[columns_to_keep], new_df[columns_to_keep]], ignore_index=True)

# # --- Normalize prices ---
# scaler = StandardScaler()
# merged_df['price_scaled'] = scaler.fit_transform(merged_df[['cena_clean']])

# --- Load scaler ---
scaler = joblib.load(SCALER_PATH)
df['price_scaled'] = scaler.transform(df[['cena_clean']])

# --- TensorFlow image decode with filtering ---
def decode_image(filename, label):
    def _load_image(filename_str, label_val):
        fname = filename_str.numpy().decode()
        path = os.path.join(IMAGE_DIR, fname)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize(IMAGE_SIZE)
                img_array = np.array(img).astype(np.float32) / 255.0
            return img_array, label_val, True
        except Exception as e:
            print(f"❌ Skipping bad image: {fname} — {e}")
            dummy = np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)
            return dummy, label_val, False

    image, label, valid = tf.py_function(
        _load_image,
        [filename, label],
        [tf.float32, tf.float32, tf.bool]
    )
    image.set_shape((*IMAGE_SIZE, 3))
    label.set_shape(())
    valid.set_shape(())
    return image, label, valid

def get_dataset(df, shuffle=False):
    filenames = df['filename'].values
    labels = df['price_scaled'].values.astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(decode_image, num_parallel_calls=AUTOTUNE)

    # Remove invalid images
    dataset = dataset.filter(lambda img, lbl, valid: valid)

    # Drop `valid` flag
    dataset = dataset.map(lambda img, lbl, _: (img, lbl), num_parallel_calls=AUTOTUNE)

    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# --- Create test dataset ---
test_ds = get_dataset(df)

# --- Load model ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Predict ---
pred_scaled = model.predict(test_ds)
true_scaled = np.concatenate([y for _, y in test_ds], axis=0)

# --- Inverse transform ---
pred_prices = scaler.inverse_transform(pred_scaled)
true_prices = scaler.inverse_transform(true_scaled.reshape(-1, 1))

# --- Evaluate ---
mae = np.mean(np.abs(pred_prices - true_prices))
print(f"📊 Mean Absolute Error: {mae:,.2f} €")

# --- Sample predictions ---
for i in range(5):
    print(f"🏠 Predicted: {pred_prices[i][0]:,.2f} €, Actual: {true_prices[i][0]:,.2f} €")

# --- Optional: Plot ---
plt.scatter(true_prices, pred_prices, alpha=0.3)
plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("Predicted vs Actual Prices")
plt.plot([true_prices.min(), true_prices.max()],
         [true_prices.min(), true_prices.max()], 'r--')  # y = x line
plt.grid()
plt.savefig('model_test.png')
plt.show()
