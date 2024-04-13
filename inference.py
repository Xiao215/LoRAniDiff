"""
Module for performing inference with a model. It supports both text-to-image and image-to-image
generation using a pre-trained model. Users can specify a prompt for text-to-image generation or
provide an input image for image-to-image transformation, along with various parameters to control
the inference process.
"""

from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPTokenizer
from ldm.ldm import LoRAniDiff

# Setup the device for model execution
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.has_mps() or torch.backends.mps.is_available():
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Initialize the tokenizer
TOKENIZER = CLIPTokenizer(
    "model_weight/vocab.json", merges_file="model_weight/merges.txt"
)

# Define file paths for the model weights
PT_FILE = "model_weight/LoRAniDiff.pt"

# Define the prompt and configuration for the generation process
PROMPT = "A pink hair girl stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
CFG_SCALE = 8  # Scale for conditional guidance, min: 1, max: 14

# Image to image configuration
INPUT_IMAGE = None
# Uncomment the following lines to enable image-to-image mode
# IMAGE_PATH = "../images/dog.jpg"
# INPUT_IMAGE = Image.open(IMAGE_PATH)

# Define the strength for the image-to-image transformation
STRENGTH = 0.9

# Load the model and set its configuration
model = LoRAniDiff(device=DEVICE, seed=42, tokenizer=TOKENIZER)
model.load_state_dict(torch.load(PT_FILE, map_location=DEVICE))

# Perform the generation
output_image = model.generate(
    caption=PROMPT, input_image=INPUT_IMAGE, strength=STRENGTH, cfg_scale=CFG_SCALE
)

# Save the generated image
SAVE_PATH = Path("image/output.jpg")
output_image.save(SAVE_PATH)

print(f"Image saved to {SAVE_PATH}")
