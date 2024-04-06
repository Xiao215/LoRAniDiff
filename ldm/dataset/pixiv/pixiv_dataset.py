import argparse
import requests
import os
from zipfile import ZipFile
from PIL import Image

def download_file(url, download_dir, filename):
    os.makedirs(download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded: {file_path}")

def unzip_file(zip_path, extract_to):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted: {zip_path} to {extract_to}")
    os.remove(zip_path)  # Remove the zip file after extraction
    print(f"Removed: {zip_path}")

def resize_images_in_directory(directory, size):
    for img_file in os.listdir(directory):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(directory, img_file)
            img = Image.open(img_path)
            img_resized = img.resize((size, size), Image.Resampling.LANCZOS)
            img_resized.save(img_path)
            print(f"Resized: {img_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and optionally resize images.")
    parser.add_argument("--resize", type=int, default=None,
                        help="Optional. Resize dimension (square) for images. If not specified, images will not be resized.")

    args = parser.parse_args()
    download_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data/pixiv')

    image_zip_url = "https://huggingface.co/datasets/Xiao215/pixiv-image-with-caption/resolve/main/pixiv_images.zip?download=true"
    parquet_url = "https://huggingface.co/datasets/Xiao215/pixiv-image-with-caption/resolve/main/pixiv_image_caption.parquet?download=true"

    download_file(image_zip_url, download_dir, "pixiv_images.zip")
    unzip_file(os.path.join(download_dir, "pixiv_images.zip"), download_dir)

    download_file(parquet_url, download_dir, "pixiv_image_caption.parquet")

    print(args.resize)
    if args.resize is not None:
        resize_images_in_directory(os.path.join(download_dir, "pixiv_images"), args.resize)
