import ldm.utils.model_loader as model_loader
import ldm.pipeline as pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from ldm.ldm import LoRAniDiff

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.has_mps or torch.backends.mps.is_available():
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer(
    "model_weight/vocab.json", merges_file="model_weight/merges.txt"
)
model_file = "model_weight/v1-5-pruned-emaonly.ckpt"
pt_file = "model_weight/LoRAniDiff.pt"


# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = "give me a image of a cat with a hat"
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

input_image = None
# Comment to disable image to image
# image_path = "../images/dog.jpg"
# input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

model = LoRAniDiff(device=DEVICE, seed=42, tokenizer=tokenizer)
model.load_state_dict(torch.load(pt_file, map_location=DEVICE))

output_image = model.generate(prompt, input_image=input_image, strength=strength)


save_path = Path("image/output.jpg")

output_image.save(save_path)

print(f"Image saved to {save_path}")
