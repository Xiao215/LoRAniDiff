import os
import pandas as pd
import random
import string
from PIL import Image
from tqdm import tqdm

# *About the DiffusionDB dataset:
# link: https://huggingface.co/datasets/poloclub/diffusiondb
# using the DiffusionDB 2M version
# 1000 images in each part, 2M total
# * for DiffusionDB, files needed in cwd: .zip of images, metadata.parquet

# Open the metadata.parquet file
metadata = pd.read_parquet("metadata.parquet")

# Create a dictionary to store image name and caption
image_dict = {}

IMAGE_SIZE = 64

IMAGE_DIR = "./part-000001"

for image_name, caption in tqdm(zip(metadata["image_name"], metadata["prompt"]), total=len(metadata)):
    # We have the saved the image
    if os.path.exists(f"{IMAGE_DIR}/{image_name}"):
        # Generate a random 10-digit number for the new image name
        new_image_name = "pixivdb" + ''.join(random.choices(string.digits, k=10)) + ".jpg"
        # Update the image name in the metadata
        metadata.loc[metadata["image_name"] == image_name, "image_name"] = new_image_name
        # Update the image name in the ./part-000001 folder
        os.rename(f"{IMAGE_DIR}/{image_name}", f"{IMAGE_DIR}/{new_image_name}")
        # Store the new image name and caption in the dictionary
        image_dict[new_image_name] = caption
        # Store the new relative image filepath in the dictionary
        metadata.loc[metadata["image_name"] == new_image_name, "image_path"] = (f"{IMAGE_DIR}{new_image_name}")


# Print the first 5 key-value pairs in image_dict
for i, (image_name, caption) in enumerate(image_dict.items()):
    print(f"Image Name: {image_name}, Caption: {caption}")
    if i == 4:
        break


# Go through all the images in the ./part-000001 folder and make them IMAGE_SIZExIMAGE_SIZE pixels
for image_name in os.listdir(IMAGE_DIR):
    # Load the image
    image = Image.open(f"./part-000001/{image_name}")
    # Resize the image to IMAGE_SIZExIMAGE_SIZE pixels
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    # Save the resized image
    image.save(f"./part-000001/{image_name}")


# Save image_dict as a parquet file
image_df = pd.DataFrame(image_dict.items(), columns=["image_name", "caption"])
image_df.to_parquet("db_images_metadata.parquet")



# Double check that all images are IMAGE_SIZExIMAGE_SIZE
for image_name in os.listdir(IMAGE_DIR):
    image = Image.open(f"{IMAGE_DIR}/{image_name}")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image.save(f"{IMAGE_DIR}/{image_name}")