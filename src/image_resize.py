from PIL import Image
import os

# Set paths
input_folder = "../data/kvadrati_images/"
output_folder = "../data/resized_images/"
target_size = (224, 224)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Batch resize
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        img_resized.save(os.path.join(output_folder, filename))

print("✅ Done resizing all images to 224x224!")
