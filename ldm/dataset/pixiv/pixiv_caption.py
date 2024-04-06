from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import os

# Specify your cache directory
cache_dir = "llava_cache/"

# Initialize the processor and model with a specified cache directory
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=cache_dir)
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=cache_dir)
model.to("cuda:0")

# Directory containing images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
image_dir = os.path.join(BASE_DIR, 'data/pixiv/images')

# Iterate through each image in the directory
for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            image = Image.open(image_path)
            prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
            inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
            output = model.generate(**inputs, max_new_tokens=100)
            caption = processor.decode(output[0], skip_special_tokens=True)
            print(f"Caption for {image_file}: {caption}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    break
