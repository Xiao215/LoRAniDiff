"""
Module for generating captions for images in the Pixiv dataset using a pre-trained LLaMA model.
This script processes images from a specified directory and generates descriptive captions using
the LLaMA model's capabilities.
"""

import os
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm
import pandas as pd

# Directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_DIR = os.path.join(BASE_DIR, "data/pixiv/images")
CACHE_DIR = os.path.join(BASE_DIR, "llava_cache/")

# Model and processor initialization
processor = LlavaNextProcessor.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=CACHE_DIR
)
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir=CACHE_DIR,
)
model.to("cuda:0")

# Caption prompt setup
PROMPT_TEMPLATE = """You are an astute observer and a skilled caption generator. \n
                     You are given an anime illusion. \n
                     Your task is to generate a descriptive caption in one sentence about what is happening, who or what is present, and any notable objects or scenery. \n
                     Describe the scene captured in this image focusing on actions and emotions. \n
                     Avoid mentioning the style. \n
                     Identify and describe the individual(s) in the scene by their apparent age, gender presentation, and any other observable traits. Use terms such as 'man', 'woman', 'girl', 'boy', or other descriptors that accurately reflect their presentation in the image.
                     Summarize in one sentence.
<image>

[INST] Based on the contents of this image, provide your caption in one sentence.[/INST] This is an anime image of"""
END_OF_PROMPT_MARKER = "[/INST] This is an anime image of"

data = []
PRINT_EXAMPLE = True
for image_file in tqdm(os.listdir(IMAGE_DIR), desc="Processing Images"):
    image_path = os.path.join(IMAGE_DIR, image_file)
    if image_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        try:
            image = Image.open(image_path)
            inputs = processor(PROMPT_TEMPLATE, image, return_tensors="pt").to("cuda:0")
            output = model.generate(**inputs, max_new_tokens=100)
            full_text = processor.decode(output[0], skip_special_tokens=True)
            parts = full_text.split(END_OF_PROMPT_MARKER)
            caption = parts[1].strip()
            if PRINT_EXAMPLE:
                print({"image_name": image_file, "caption": caption})
                PRINT_EXAMPLE = False
            data.append({"image_name": image_file, "caption": caption})
        except FileNotFoundError as e:
            print(f"Error processing {image_file}: {e}")

df = pd.DataFrame(data)
PARQUET_PATH = "/content/pixiv_image_caption.parquet"
os.makedirs(os.path.dirname(PARQUET_PATH), exist_ok=True)
df.to_parquet(PARQUET_PATH, index=False)

# Loading and displaying the dataframe
df_loaded = pd.read_parquet(PARQUET_PATH)
print(f"Number of entries in the Parquet file: {len(df_loaded)}")
print("Last few entries in the Parquet file:")
print(df_loaded.tail())
