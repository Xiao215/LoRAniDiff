import os
import urllib.request
import zipfile

# URLs of the files to download
json_url = "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json"
zip_url = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"

# Paths for saving files
json_file = "metadata.json"
zip_file = "images.zip"
target_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data/textcaps')

# Ensure the target directory exists, create if it does not
os.makedirs(target_data_dir, exist_ok=True)

# Check if JSON file already exists
if not os.path.exists(os.path.join(target_data_dir, json_file)):
    # Download JSON file
    urllib.request.urlretrieve(json_url, os.path.join(target_data_dir, json_file))

# Check if the 'train' folder already exists instead of 'train_image'
if not os.path.exists(os.path.join(target_data_dir, 'train')):
    # Process for handling the ZIP file
    # Check if ZIP file already exists
    if not os.path.exists(os.path.join(target_data_dir, zip_file)):
        # Download ZIP file
        urllib.request.urlretrieve(zip_url, os.path.join(target_data_dir, zip_file))

    # Extract ZIP file
    with zipfile.ZipFile(os.path.join(target_data_dir, zip_file), 'r') as zip_ref:
        zip_ref.extractall(target_data_dir)

    # Clean up: Remove downloaded ZIP file after extraction
    os.remove(os.path.join(target_data_dir, zip_file))

    # Rename the 'train_images' folder to 'train'
    os.rename(os.path.join(target_data_dir, 'train_images'), os.path.join(target_data_dir, 'train'))
else:
    print("The 'train' folder already exists. Skipping image download, extraction, and renaming.")
