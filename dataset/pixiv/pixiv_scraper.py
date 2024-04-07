"""
Module for scraping images from Pixiv. This script downloads images given their URLs,
provided they match a specified target prefix. The images are saved to a designated directory,
avoiding downloads of already existing files.
"""

import os
import requests  # Moved below standard import 'os' as per PEP 8


def download_image_from_url(url: str, download_dir: str) -> None:
    """
    Downloads an image from a given URL into the specified directory.
    Args:
        url (str): The URL of the image to download.
        download_dir (str): The directory to save the downloaded image.
    """
    # Extract the image ID and construct the filename
    image_id = url.split("/")[-1].split("_")[0]  # Extract the image ID part
    filename = os.path.join(download_dir, f"pixiv{image_id}.jpg")

    # Check if the file already exists to avoid re-downloading
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return

    # Set the referer header required by Pixiv
    referer_header = {"Referer": "https://www.pixiv.net/"}

    # Downloading the image
    response = requests.get(url, headers=referer_header)
    if response.status_code == 200:
        with open(filename, "wb") as img_file:
            img_file.write(response.content)
        print(f"Downloaded {url} to {filename}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")


def prompt_for_urls(directory: str, prefix: str) -> None:
    """
    Prompts the user to input image URLs for downloading, until 'done' is entered.
    Only downloads images that match the specified prefix.
    Args:
        directory (str): The directory where images will be downloaded.
        prefix (str): The required prefix for the image URLs.
    """
    while True:
        print("Enter image URLs (enter 'done' to finish):")
        url = input().strip()
        if url.lower() == "done":
            break
        if url.startswith(prefix):
            download_image_from_url(url, directory)
        else:
            print("URL does not match the target prefix. Skipping.")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    IMAGE_DIR = os.path.join(BASE_DIR, "data/pixiv/images")
    TARGET_PREFIX = "https://i.pximg.net/c/360x360_70"

    # Ensure the image directory exists
    os.makedirs(IMAGE_DIR, exist_ok=True)

    prompt_for_urls(IMAGE_DIR, TARGET_PREFIX)

    # After downloading, count the number of image files
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
    print(f"Total images in '{IMAGE_DIR}': {len(image_files)}")
