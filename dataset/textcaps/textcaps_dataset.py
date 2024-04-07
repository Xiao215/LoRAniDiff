"""
This module is responsible for downloading and preparing the TextCaps dataset for processing.
It downloads a JSON metadata file and a ZIP file containing images, ensuring that the data
is organized in the expected directory structure for further use.
"""

import os
import urllib.request
import zipfile

# URLs of the files to download
JSON_URL = (
    "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json"
)
ZIP_URL = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"

# Paths for saving files
JSON_FILE = "metadata.json"
ZIP_FILE = "images.zip"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET_DATA_DIR = os.path.join(BASE_DIR, "data/textcaps")

# Ensure the target directory exists, create if it does not
os.makedirs(TARGET_DATA_DIR, exist_ok=True)

# Check if JSON file already exists
if not os.path.exists(os.path.join(TARGET_DATA_DIR, JSON_FILE)):
    # Download JSON file
    urllib.request.urlretrieve(JSON_URL, os.path.join(TARGET_DATA_DIR, JSON_FILE))

# Check if the 'train' folder already exists instead of 'train_image'
if not os.path.exists(os.path.join(TARGET_DATA_DIR, "train")):
    # Process for handling the ZIP file
    # Check if ZIP file already exists
    if not os.path.exists(os.path.join(TARGET_DATA_DIR, ZIP_FILE)):
        # Download ZIP file
        urllib.request.urlretrieve(ZIP_URL, os.path.join(TARGET_DATA_DIR, ZIP_FILE))

    # Extract ZIP file
    with zipfile.ZipFile(os.path.join(TARGET_DATA_DIR, ZIP_FILE), "r") as zip_ref:
        zip_ref.extractall(TARGET_DATA_DIR)

    # Clean up: Remove downloaded ZIP file after extraction
    os.remove(os.path.join(TARGET_DATA_DIR, ZIP_FILE))

    # Rename the 'train_images' folder to 'train'
    os.rename(
        os.path.join(TARGET_DATA_DIR, "train_images"),
        os.path.join(TARGET_DATA_DIR, "train"),
    )
else:
    print(
        "The 'train' folder already exists. Skipping image download, extraction, and renaming."
    )
