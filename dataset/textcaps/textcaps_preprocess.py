"""
This module preprocesses the TextCaps dataset by resizing images to a specified size,
padding them to maintain aspect ratio, and generating a simplified metadata file.
It processes a set number of images and captions, then saves the modified images and
new metadata to a Parquet file for efficient access.
"""

import json
import os
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import random

# Correct the order of imports
import shutil

# Constants and Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_DIR = os.path.join(BASE_DIR, "data/textcaps/train")
METADATA_PATH = os.path.join(BASE_DIR, "data/textcaps/metadata.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/textcaps/processed")
PARQUET_PATH = os.path.join(BASE_DIR, "data/textcaps/metadata.parquet")
IMAGE_SIZE = 256  # Corrected to UPPER_CASE

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pad_and_resize_image(image, target_size):
    """
    Pads and resizes an image to a square of the target size.
    Args:
        image: The PIL Image to resize.
        target_size: The target size of the image's width and height.
    Returns:
        The resized and padded image as a PIL Image.
    """
    # Calculate padding
    max_size = max(image.size)
    padding = (max_size - image.width, max_size - image.height)
    # Pad the image to make it square
    padded_image = ImageOps.expand(image, (0, 0, padding[0], padding[1]), fill="black")
    # Resize the padded image
    return padded_image.resize((target_size, target_size))


# Load the metadata with specified encoding
with open(METADATA_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)["data"]

simplified_data = []
visited = set()

for i, item in tqdm(enumerate(data), total=len(data)):
    if i == 1000:
        break
    image_id = item["image_id"]
    if image_id in visited:
        continue

    new_image_name = f"image_{i:06}.jpg"
    new_image_path = os.path.join(OUTPUT_DIR, new_image_name)

    img_path = os.path.join(IMAGE_DIR, item["image_path"])
    with Image.open(img_path) as img:
        processed_img = pad_and_resize_image(img, IMAGE_SIZE)
        processed_img.save(new_image_path)

    simplified_item = {
        "image_name": new_image_name,
        "image_path": new_image_path,
        "caption": random.choice(item.get("reference_strs", [""])),
    }

    simplified_data.append(simplified_item)
    visited.add(image_id)

df_simplified = pd.DataFrame(simplified_data)
df_simplified.to_parquet(PARQUET_PATH)

print("Processing complete.")

# Cleanup section
if os.path.exists(IMAGE_DIR):
    shutil.rmtree(IMAGE_DIR)
    print(f"Removed directory: {IMAGE_DIR}")

if os.path.exists(METADATA_PATH):
    os.remove(METADATA_PATH)
    print(f"Removed file: {METADATA_PATH}")

print("Cleanup successful.")
