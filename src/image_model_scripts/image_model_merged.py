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
IMAGE_DIR = "../data/kvadrati_images/"
CSV_PATH = "../data/kvadrati2/kvadrati_normalized.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# --- Load and preprocess CSV ---
df = pd.read_csv(CSV_PATH)
df = df[(df['vrsta'] == "Stanovanje") | (df['vrsta'] == "Hiša")]
df['filename'] = df['id'].astype(str) + '.jpg'

# nepremicnine dataset
new_csv_path = "../data/nepremicnine_prodaja_slike.csv"

new_df = pd.read_csv(new_csv_path)
new_df = new_df.dropna(subset=['price', 'id']).copy()

# Add filename column
new_df['filename'] = new_df['id'].astype(str) + '.jpg'
new_df = new_df[(new_df['price'] > 10_000) & (new_df['price'] < 10_000_000)]

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
df = df[(df["cena_clean"].notnull()) & (df["cena_clean"] < 10_000_000)  & (df["cena_clean"] > 10_000)]
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
new_df = new_df[new_df['filename'].apply(is_valid_image_file)].reset_index(drop=True)

columns_to_keep = ['id', 'filename', 'cena_clean']
merged_df = pd.concat([df[columns_to_keep], new_df[columns_to_keep]], ignore_index=True)
################# REMOVE THIS ################
merged_df = df

def get_velikost(v):
    split_v = str(v).split()
    try:
        return float(split_v[0].replace(',', '.'))
    except ValueError:
        return None

merged_df['velikost_clean'] = merged_df['velikost'].apply(get_velikost)
merged_df['cena_na_m2'] = merged_df['cena_clean'] / merged_df['velikost_clean']

# --- Normalize prices ---
scaler = StandardScaler()
merged_df['price_scaled'] = scaler.fit_transform(merged_df[['cena_na_m2']])

# print("Top 10 highest prices with IDs:")
# print(df.nlargest(10, 'cena_clean')[['id', 'cena_clean']])

# print("\nTop 10 lowest prices with IDs:")
# print(df.nsmallest(20, 'cena_clean')[['id', 'cena_clean']])


# --- Split dataset ---
train_df, val_df = train_test_split(merged_df, test_size=0.2, random_state=42)

# --- TensorFlow data pipeline ---
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
train_ds = train_ds.repeat()
val_ds = get_dataset(val_df, shuffle=False)

# --- Build model ---
print("Creating model...")
base_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3)
)
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Only keep last 20 layers trainable
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.Huber(),
              metrics=['mae'])

# --- Save model architecture early ---
model.save("../models/real_estate_model.keras")

# --- Train model ---
steps_per_epoch = len(train_df) // BATCH_SIZE

history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=len(val_df) // BATCH_SIZE,
    epochs=10,
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
sum_napaka = 0
for i in range(pred_prices.shape[0]):
    print(f"Predicted: {pred_prices[i][0]:,.2f}, Actual: {true_prices[i][0]:,.2f}")
    
print("Število testnih podatkov:", pred_prices.shape[0])
stevec = 0
for i in range(pred_prices.shape[0]):
    sum_napaka += abs(pred_prices[i][0] - true_prices[i][0])
    stevec += 1
    print("Napaka:", abs(pred_prices[i][0] - true_prices[i][0]))

print("Povprečna napaka:", sum_napaka/stevec, sum_napaka, stevec)

# plot errors
plt.scatter(true_prices, pred_prices, alpha=0.3)
plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("Predicted vs Actual Prices")
plt.plot([true_prices.min(), true_prices.max()],
         [true_prices.min(), true_prices.max()], 'r--')  # y = x line
plt.grid()
plt.savefig('model_validation.png')
plt.show()

# --- Save scaler ---
joblib.dump(scaler, "../models/price_scaler.pkl")