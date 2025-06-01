import os
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import joblib

# --- GPU Memory Setup ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# --- Paths ---
IMAGE_DIR = "../data/kvadrati_images/"
CSV_PATH = "../data/kvadrati2/kvadrati_normalized.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# --- Load DataFrame ---
df = pd.read_csv(CSV_PATH)
df = df[(df['vrsta'] == "Stanovanje") | (df['vrsta'] == "Hiša")]
df['filename'] = df['id'].astype(str) + '.jpg'

# --- Price Cleaning ---
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

# --- Remove invalid/corrupt images ---
def is_valid_image(fname):
    path = os.path.join(IMAGE_DIR, fname)
    if not os.path.exists(path):
        print(f"Missing: {fname}")
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Corrupt: {fname} — {e}")
        return False

df = df[df['filename'].apply(is_valid_image)].reset_index(drop=True)

# --- Scale Target Price ---
price_scaler = StandardScaler()
df['price_scaled'] = price_scaler.fit_transform(df[['cena_clean']])

# --- Feature Selection ---
num_features = ['velikost_clean', 'leto_gradnje']
cat_features = ['prodajalec_agencija', 'Ener. izk.', 'tip']
bin_features = [col for col in df.columns if col.startswith(('Št.', 'Dodatno_', 'Lega_', 'Luksuz_', 'Ogrevanje_', 'Priključki_'))]

structured_features = num_features + cat_features + bin_features
df = df.dropna(subset=structured_features).reset_index(drop=True)

# --- Preprocessing Pipelines ---
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features),
    ("bin", "passthrough", bin_features)
])

X_structured = preprocessor.fit_transform(df[structured_features]).astype(np.float32)
joblib.dump(preprocessor, "../models/preprocessor.pkl")

# --- Split Data ---
train_df, val_df, X_structured_train, X_structured_val, y_train, y_val = train_test_split(
    df, X_structured, df['price_scaled'].values.astype(np.float32),
    test_size=0.2, random_state=42
)

# --- TensorFlow Dataset ---
def decode_image(filename, label, structured):
    def _decode(filename_str, label_val, structured_np):
        try:
            path = os.path.join(IMAGE_DIR, filename_str.numpy().decode())
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, IMAGE_SIZE)
            image = tf.cast(image, tf.float32) / 255.0
            return image, structured_np, label_val, True
        except Exception as e:
            print(f"Error decoding {filename_str.numpy().decode()} — {e}")
            dummy_img = np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)
            return dummy_img, structured_np, label_val, False

    image, struct, label, valid = tf.py_function(
        _decode, [filename, label, structured],
        [tf.float32, tf.float32, tf.float32, tf.bool]
    )
    image.set_shape((*IMAGE_SIZE, 3))
    struct.set_shape([X_structured.shape[1]])
    label.set_shape(())
    valid.set_shape(())
    return (image, struct), label, valid

def get_dataset(df, X_structured, y, shuffle=True):
    filenames = df['filename'].values
    dataset = tf.data.Dataset.from_tensor_slices((filenames, y, X_structured))
    dataset = dataset.map(decode_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.filter(lambda x, y, valid: valid)
    dataset = dataset.map(lambda x, y, _: (x, y), num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = get_dataset(train_df, X_structured_train, y_train)
val_ds = get_dataset(val_df, X_structured_val, y_val, shuffle=False)

# --- Model Architecture ---
print("Creating model...")
img_input = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image_input")
struct_input = tf.keras.Input(shape=(X_structured.shape[1],), name="struct_input")

base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=img_input)
base_model.trainable = False

x_img = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x_struct = tf.keras.layers.Dense(64, activation='relu')(struct_input)

x = tf.keras.layers.concatenate([x_img, x_struct])
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=[img_input, struct_input], outputs=output)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- Train Model ---
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# --- Plot Training ---
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.legend()
plt.title('Mean Absolute Error over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.savefig('training_plot.png')
plt.show()

# --- Save Model and Scaler ---
model.save("../models/multi_input_model.keras")
joblib.dump(price_scaler, "../models/price_scaler.pkl")

# --- Inference Example ---
print("Evaluating predictions...")
y_pred_scaled = model.predict(val_ds)
y_true_scaled = np.concatenate([y for _, y in val_ds], axis=0)
y_pred = price_scaler.inverse_transform(y_pred_scaled)
y_true = price_scaler.inverse_transform(y_true_scaled.reshape(-1, 1))

for i in range(5):
    print(f"Predicted: {y_pred[i][0]:,.2f}, Actual: {y_true[i][0]:,.2f}")
