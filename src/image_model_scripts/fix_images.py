import os
from PIL import Image

# Define the image directory
image_dir = "../data/kvadrati_new_images"
fixed_dir = "../data/fixed_kvadrati_new_images"

# Create directory for fixed images
os.makedirs(fixed_dir, exist_ok=True)

# Supported extension (optional: you could check .jpeg too)
valid_extensions = {".jpg", ".jpeg"}

for filename in os.listdir(image_dir):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in valid_extensions:
        continue

    input_path = os.path.join(image_dir, filename)
    output_path = os.path.join(fixed_dir, filename)

    try:
        with Image.open(input_path) as img:
            # Convert to RGB to ensure compatibility (some JPEGs are grayscale or CMYK)
            rgb_img = img.convert("RGB")
            rgb_img.save(output_path, "JPEG", quality=95)
            print(f"Fixed: {filename}")
    except Exception as e:
        print(f"Failed to fix {filename}: {e}")
