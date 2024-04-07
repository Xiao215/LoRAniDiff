from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd

# Directory containing images
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))))
image_dir = os.path.join(BASE_DIR, "data/pixiv/images")
# Specify your cache directory
cache_dir = os.path.join(BASE_DIR, "llava_cache/")

# Initialize the processor and model with a specified cache directory
processor = LlavaNextProcessor.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=cache_dir
)
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir=cache_dir,
)
model.to("cuda:0")

image_dir = "/content/drive/My Drive/aaa"

prompt = """You are an astute observer and a skilled caption generator. \n
            You are given an anime illusion. \n
            Your task is to generate a detail descriptive caption in one sentence about what is happening, who or what is present, and any notable objects or scenery. \n
            Describe the scene captured in this image focusing on actions and emotions. \n
            Avoid mentioning the style. \n
            Identify and describe the individual(s) in the scene by their apparent age, gender presentation, and any other observable traits. Use terms such as 'man', 'woman', 'girl', 'boy', or other descriptors that accurately reflect their presentation in the image.
            Summarize in one sentence.
<image>

[INST] Based on the contents of this image, provide your caption in one sentence.[/INST] This is an anime image of"""
end_of_prompt_marker = "[/INST] This is an anime image of"

data = []
print_example = True
for image_file in tqdm(os.listdir(image_dir), desc="Processing Images"):
    image_path = os.path.join(image_dir, image_file)
    if image_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        try:
            image = Image.open(image_path)
            # display(image)

            prompt = prompt

            inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
            output = model.generate(**inputs, max_new_tokens=100)
            full_text = processor.decode(output[0], skip_special_tokens=True)
            parts = full_text.split(end_of_prompt_marker)
            caption = parts[1].strip()
            # print(f"Caption for {image_file}: {caption}")
            if print_example:
                print_example = False
                print({"image_name": image_file, "caption": caption})
            data.append({"image_name": image_file, "caption": caption})
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

df = pd.DataFrame(data)
parquet_path = "/content/pixiv_image_caption.parquet"
directory = os.path.dirname(parquet_path)
os.makedirs(directory, exist_ok=True)
df.to_parquet(parquet_path, index=False)
df = pd.read_parquet(parquet_path)

# Printing the number of entries
print(f"Number of entries in the Parquet file: {len(df)}")

# Displaying the last few entries
print("Last few entries in the Parquet file:")
print(df.tail())
