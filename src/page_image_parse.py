import os
import re
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

input_dir = '../data/kvadrati_new/pages'
output_dir = '../data/kvadrati_new/images'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Regex pattern to match file names like html_vsebina_1
pattern = re.compile(r'html_vsebina_(\d+)\.html$')

for filename in os.listdir(input_dir):
    match = pattern.match(filename)
    if not match:
        continue

    file_number = match.group(1)
    file_path = os.path.join(input_dir, filename)

    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Find the first img with class 'pzl-gallery-item'
    img_tag = soup.find('img', class_='pzl-gallery-item')
    if not img_tag or not img_tag.get('src'):
        print(f"No matching image found in {filename}")
        continue

    img_url = img_tag['src']
    try:
        response = requests.get(img_url)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        img_path = os.path.join(output_dir, f"{file_number}.jpg")

        # Convert to RGB if not already JPEG-compatible (e.g. PNG with alpha)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img.save(img_path, format="JPEG")
        print(f"Saved valid JPEG image for {filename} as {file_number}.jpg")

    except Exception as e:
        print(f"Failed to process image from {img_url} in {filename}: {e}")
