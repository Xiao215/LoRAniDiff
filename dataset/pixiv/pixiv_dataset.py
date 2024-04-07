"""
This module provides functionality to download a dataset of images and their corresponding captions
from Pixiv, optionally resize the images, and prepare the dataset for further processing or analysis.
It handles downloading zip files, extracting them, and resizing images to a specified size.
"""

import argparse
import os
import requests
from zipfile import ZipFile
from PIL import Image


def download_file(url: str, download_dir: str, filename: str):
    """
    Downloads a file from a given URL into the specified directory with the specified filename.
    Args:
        url: The URL from which to download the file.
        download_dir: The directory to save the downloaded file.
        filename: The name of the file to save the download as.
    """
    os.makedirs(download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded: {file_path}")


def unzip_file(zip_path: str, extract_to: str):
    """
    Extracts a zip file to a specified directory.
    Args:
        zip_path: The path to the zip file to extract.
        extract_to: The directory to extract the zip file contents into.
    """
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted: {zip_path} to {extract_to}")
    os.remove(zip_path)  # Remove the zip file after extraction
    print(f"Removed: {zip_path}")


def resize_images_in_directory(directory: str, size: int):
    """
    Resizes all images in a directory to a specified size.
    Args:
        directory: The directory containing images to resize.
        size: The new size (width and height) to resize the images to.
    """
    for img_file in os.listdir(directory):
        if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            img_path = os.path.join(directory, img_file)
            img = Image.open(img_path)
            img_resized = img.resize((size, size), Image.LANCZOS)
            img_resized.save(img_path)
            print(f"Resized: {img_file} to size {size} x {size}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and optionally resize images."
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional. Resize dimension (square) for images. If not specified, images will not be resized.",
    )

    args = parser.parse_args()
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    DOWNLOAD_DIR = os.path.join(BASE_DIR, "data/pixiv")

    IMAGE_ZIP_URL = "https://huggingface.co/datasets/Xiao215/pixiv-image-with-caption/resolve/main/pixiv_images.zip?download=true"
    PARQUET_URL = "https://huggingface.co/datasets/Xiao215/pixiv-image-with-caption/resolve/main/pixiv_image_caption.parquet?download=true"

    download_file(IMAGE_ZIP_URL, DOWNLOAD_DIR, "pixiv_images.zip")
    unzip_file(os.path.join(DOWNLOAD_DIR, "pixiv_images.zip"), DOWNLOAD_DIR)
    download_file(PARQUET_URL, DOWNLOAD_DIR, "pixiv_image_caption.parquet")

    if args.resize is not None:
        resize_images_in_directory(
            os.path.join(DOWNLOAD_DIR, "pixiv_images"), args.resize
        )
