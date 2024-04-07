import torch
from torchvision import transforms
import os
from transformers import CLIPTokenizer
from ldm.ldm import LoRAniDiff

# Device configuration
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load pre-initialized models
tokenizer = CLIPTokenizer(
    "model_weight/vocab.json", merges_file="model_weight/merges.txt"
)
model_file = "model_weight/v1-5-pruned-emaonly.ckpt"

# Initialize the model
model = LoRAniDiff(device, model_file=model_file).to(device)

# If multiple GPUs are available, use DataParallel
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

# Move model to the configured device
model.to(device)

# Load the pre-trained weights
# Assuming `LoRAniDiff`'s constructor automatically loads the weights from `model_file`.
# If not, you might need a model.load_state_dict(torch.load(model_file)) statement here, adjusted as per your model's architecture.

# Model saving path
model_save_path = "model_weight/"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path, exist_ok=True)

# Saving the model weights
final_save_path = os.path.join(model_save_path, "LoRAniDiff_final_model.pt")
torch.save(model.state_dict(), final_save_path)
print(f"Model weights saved to {final_save_path}")
