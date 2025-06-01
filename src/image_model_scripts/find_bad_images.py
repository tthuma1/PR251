import os
import tensorflow as tf
from PIL import Image

# Set your image directory path
IMAGE_DIR = "../data/kvadrati_images/"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # Supported formats

def is_valid_image_tf(image_path):
    try:
        image_bytes = tf.io.read_file(image_path)
        tf.image.decode_image(image_bytes, channels=3)  # Automatically detects format
        return True
    except Exception as e:
        new_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'fixed_images', os.path.basename(image_path))
        repair_image(image_path, new_path)
        return False

def repair_image(image_path, output_path=None):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')  # Ensure RGB format
            if not output_path:
                output_path = image_path  # Overwrite original
            img.save(output_path, format='JPEG', quality=95)
        return True
    except Exception as e:
        print(f"❌ Cannot repair {image_path}: {e}")
        return False

def find_corrupt_images(image_dir):
    bad_images = []

    for fname in os.listdir(image_dir):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            path = os.path.join(image_dir, fname)
            if not is_valid_image_tf(path):
                bad_images.append(fname)

    return bad_images

if __name__ == "__main__":
    print(f"Scanning for corrupt images in: {IMAGE_DIR}")
    bad_files = find_corrupt_images(IMAGE_DIR)

    if not bad_files:
        print("✅ All images are valid.")
    else:
        print(f"❌ Found {len(bad_files)} corrupt/unreadable image(s):")
        for bad in bad_files:
            print(f" - {bad}")
