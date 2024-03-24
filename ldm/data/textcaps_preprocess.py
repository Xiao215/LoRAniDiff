import os
import json
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import random
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_DIR = os.path.join(BASE_DIR, 'data/textcaps/train')
METADATA_PATH = os.path.join(BASE_DIR, 'data/textcaps/metadata.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/textcaps/processed')
PARQUET_PATH = os.path.join(BASE_DIR, 'data/textcaps/metadata.parquet')

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to pad and resize image
def pad_and_resize_image(image, target_size):
    # Calculate padding
    max_size = max(image.size)
    padding = (max_size - image.width, max_size - image.height)
    # Pad the image to make it square
    padded_image = ImageOps.expand(image, (0, 0, padding[0], padding[1]), fill='black')
    # Resize the padded image
    return padded_image.resize((target_size, target_size))

# Load the metadata
with open(METADATA_PATH, 'r') as file:
    data = json.load(file)['data']

# Process each image and update metadata
visited_images = set()  # Set to store visited image IDs
# Assuming the beginning of your script is unchanged...

# New list to hold simplified metadata
simplified_data = []

for i, item in tqdm(enumerate(data), total=len(data)):
    image_id = item['image_id']
    # Check if the image has already been visited
    if image_id in visited_images:
        continue  # Skip this image

    # New image name and path
    new_image_name = f"image_{i:06}.jpg"
    new_image_path = os.path.join(OUTPUT_DIR,  item['image_id']+'.jpg')

    # Load, pad, and resize the image
    img_path = os.path.join(BASE_DIR, 'data/textcaps/'+item['image_path'])
    with Image.open(img_path) as img:
        processed_img = pad_and_resize_image(img, 512)  # Assuming 512x512 as target size
        processed_img.save(new_image_path)

    # Create simplified metadata entry
    simplified_item = {
        'image_name': new_image_name,
        'image_path': new_image_path,
        'caption': random.choice(item.get('reference_strs', [""]))  # Select one random caption
    }

    # Add to the simplified data list
    simplified_data.append(simplified_item)

    # Mark this image ID as visited
    visited_images.add(image_id)

# Convert simplified metadata to DataFrame and save as Parquet
df_simplified = pd.DataFrame(simplified_data)
df_simplified.to_parquet(PARQUET_PATH)

print("Processing complete.")

# if os.path.exists(IMAGE_DIR):
#     shutil.rmtree(IMAGE_DIR)
#     print(f"Removed directory: {IMAGE_DIR}")
# else:
#     print(f"Directory {IMAGE_DIR} not found or already removed.")

if os.path.exists(METADATA_PATH):
    os.remove(METADATA_PATH)
    print(f"Removed file: {METADATA_PATH}")
else:
    print(f"File {METADATA_PATH} not found or already removed.")

print('Cleanup successful.')
