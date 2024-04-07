import requests
import os


def download_image_from_url(url, download_dir):
    # Extract the image ID and construct the filename
    image_id = url.split("/")[-1].split("_")[0]  # Extract the image ID part
    filename = os.path.join(download_dir, f"pixiv{image_id}.jpg")

    # Check if the file already exists to avoid re-downloading
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return

    # Set the referer header required by pixiv
    referer_header = {"Referer": "https://www.pixiv.net/"}

    # Downloading the image
    response = requests.get(url, headers=referer_header)
    if response.status_code == 200:
        with open(filename, "wb") as img_file:
            img_file.write(response.content)
        print(f"Downloaded {url} to {filename}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")


def prompt_for_urls(image_dir, target_prefix):
    while True:
        print("Enter image URLs (enter 'done' to finish):")
        url = input().strip()
        if url.lower() == "done":
            break
        if url.startswith(target_prefix):
            download_image_from_url(url, image_dir)
        else:
            print("URL does not match the target prefix. Skipping.")


def main():
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    image_dir = os.path.join(BASE_DIR, "data/pixiv/images")

    # Specify the target prefix
    target_prefix = "https://i.pximg.net/c/360x360_70"

    # Ensure the image directory exists
    os.makedirs(image_dir, exist_ok=True)

    prompt_for_urls(image_dir, target_prefix)

    # After downloading, count the number of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    print(f"Total images in '{image_dir}': {len(image_files)}")


if __name__ == "__main__":
    main()
