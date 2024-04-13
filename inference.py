"""
This module performs inference using a pre-trained model to generate images from textual prompts.
It supports both text-to-image and image-to-image generation, with options to configure the generation process.
"""

from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPTokenizer
from ldm.utils import model_loader
from ldm import pipeline

# Setup device for model inference
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.has_mps() or torch.backends.mps.is_available():
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Load tokenizer and models
TOKENIZER = CLIPTokenizer(
    "model_weight/vocab.json", merges_file="model_weight/merges.txt"
)
MODEL_FILE = "model_weight/v1-5-pruned-emaonly.ckpt"
MODELS = model_loader.preload_models_from_standard_weights(MODEL_FILE, DEVICE)




# Setup for text-to-image generation
# PROMPT = "An anime girl with white hair and a black and gold outfit, soaring through the air with a bird on her shoulder, surrounded by autumn leaves and a wooden bridge, highly detailed, ultra sharp, 8k resolution"
PROMPT = "Same dog but sitting on a beach, highly detailed, ultra sharp, 8k resolution"
UNCOND_PROMPT = ""  # Also known as negative prompt
DO_CFG = True
CFG_SCALE = 8  # Min: 1, Max: 14

# Setup for image-to-image generation
INPUT_IMAGE = None
IMAGE_PATH = Path("image/image-image/dog_og.png")
# Uncomment the following line to enable image-to-image mode
INPUT_IMAGE = Image.open(IMAGE_PATH)
STRENGTH = 0.6  # Control the deviation from the input image

# Setup for the sampling process
SAMPLER = "ddpm"
NUM_INFERENCE_STEPS = 50
SEED = 42

# Generate the image
output_image = pipeline.generate(
    prompt=PROMPT,
    uncond_prompt=UNCOND_PROMPT,
    input_image=INPUT_IMAGE,
    strength=STRENGTH,
    do_cfg=DO_CFG,
    cfg_scale=CFG_SCALE,
    sampler_name=SAMPLER,
    n_inference_steps=NUM_INFERENCE_STEPS,
    seed=SEED,
    models=MODELS,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=TOKENIZER,
)
# Process and save the output image
image = Image.fromarray(output_image)
SAVE_DIR = Path("image")  # Define the directory to save the image
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = Path("image/output.jpg")
image.save(SAVE_PATH)

print(f"Image saved to {SAVE_PATH}")
