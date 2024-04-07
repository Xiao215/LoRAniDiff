import os
import requests

# URLs of the files you want to download and their corresponding file names
files_to_download = [
    ("https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt?download=true",
     "merges.txt",
     ),
    ("https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json?download=true",
     "vocab.json",
     ),
    ("https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt?download=true",
     "v1-5-pruned-emaonly.ckpt",
     ),
    ("https://huggingface.co/Xiao215/LoRAniDiff/resolve/main/LoRAniDiff.pt?download=true",
     "LoRAniDiff.pt",
     ),
]

# Base directory where you want to save the files
base_directory = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))),
    "model_weight",
)

# Ensure the base directory exists, if not, create it
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

# Loop through each file to download
for url, filename in files_to_download:
    # Full path for the file to be saved
    file_path = os.path.join(base_directory, filename)

    # Make a GET request to download the file
    response = requests.get(url)

    # Ensure the request was successful
    if response.status_code == 200:
        # Open the file in write mode ('wb' is write binary mode) and save the
        # content
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"File has been downloaded and saved to {file_path}")
    else:
        print(
            f"Failed to download {filename}. Status code: {response.status_code}")
