import requests
import os

def download_image_from_url(url, download_dir, referer_header):
    # Extracting a unique part from each URL to use as filename
    unique_part = url.split('/')[-1]
    filename = f"{download_dir}/{unique_part}.jpg"

    # Check if the file already exists to avoid re-downloading
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return

    # Downloading the image
    response = requests.get(url, headers=referer_header)
    if response.status_code == 200:
        with open(filename, 'wb') as img_file:
            img_file.write(response.content)
        print(f"Downloaded {url} to {filename}")
    else:
        print(f"Failed to download {url}")

def process_txt_file(txt_file_path, target_prefix, referer_header, download_dir):
    with open(txt_file_path, "r") as file:
        for line in file:
            url = line.strip()
            if url.startswith(target_prefix):
                download_image_from_url(url, download_dir, referer_header)

def main():
    txt_dir = "/h/u6/c4/05/zha11021/CSC413/Stable-Diffusion/data/pixiv_url"
    target_prefix = "https://i.pximg.net/c/360x360_70"
    referer_header = {"Referer": "https://www.pixiv.net/"}
    download_dir = "downloads"

    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # List all .txt files in the txt_dir
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    for txt_file in txt_files:
        txt_file_path = os.path.join(txt_dir, txt_file)
        print(f"Processing file: {txt_file_path}")
        process_txt_file(txt_file_path, target_prefix, referer_header, download_dir)

    # After processing all files, count the number of image files
    image_files = [f for f in os.listdir(download_dir) if f.endswith('.jpg')]
    print(f"Total images in '{download_dir}': {len(image_files)}")


if __name__ == "__main__":
    main()
