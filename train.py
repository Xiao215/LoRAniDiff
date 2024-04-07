import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
from transformers import CLIPTokenizer
import pandas as pd
from PIL import Image
from ldm.ldm import LoRAniDiff
from torch.utils.data.dataset import random_split

WIDTH = 512
HEIGHT = 512

# Hyperparameters
learning_rate = 1e-4
batch_size = 2
epochs = 10
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
alpha = 0.5  # Weight between reconstruction and CLIP-guided losses

seed = 1337
generator = torch.Generator(device=device)
generator.manual_seed(seed)


# Load pre-initialized models
tokenizer = CLIPTokenizer(
    "model_weight/vocab.json", merges_file="model_weight/merges.txt"
)
model_file = "model_weight/v1-5-pruned-emaonly.ckpt"
pt_file = "model_weight/LoRAniDiff.pt"


# Prepare dataset
from torch.utils.data import Dataset, DataLoader


class TextCapsDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        """
        Args:
            parquet_file (string): Path to the metadata parquet file.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata_df = pd.read_parquet(parquet_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.metadata_df.iloc[idx]["image_path"]
        image = Image.open(img_path).convert("RGB")
        caption = self.metadata_df.iloc[idx]["caption"]
        print(f"caption: {caption}")
        print(f"image: {image.size}")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "caption": caption}


# Define your transformations, e.g., ToTensor
transform = transforms.Compose(
    [
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
    ]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(BASE_DIR, "data/textcaps/metadata.parquet")
dataset = TextCapsDataset(parquet_path, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Use DataLoader to handle batching
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

model = LoRAniDiff(device, model_file=model_file, width=WIDTH, height=HEIGHT).to(device)
model.load_state_dict(torch.load(pt_file, map_location=device))
print(f"Number of GPU is: {torch.cuda.device_count()}")
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)


optimizer = Adam(model.parameters(), lr=learning_rate)


def evaluate_model(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No need to track gradients for validation
        for batch in data_loader:
            images = batch["image"].to(device)
            captions = batch["caption"]
            # The following line might need adjustments based on your model's specifics
            generated_images, text_embeddings = model(images, captions, tokenizer)
            loss = model.compute_loss(generated_images, images, text_embeddings)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss


print("Training...")
accumulation_steps = 0
for epoch in range(epochs):
    model.train()  # Set model to training mode
    total_loss = 0

    for i, batch in enumerate(tqdm(train_loader)):
        print(f"iter: {i}")
        images = batch["image"].to(device)
        captions = batch["caption"]
        uncond_prompt = [""] * len(captions)
        print(f"images: {images.size()}")
        print(f"captions: {captions}")
        generated_images, text_embeddings, _ = model(images, captions, uncond_prompt)

        print(f"generated_images: {generated_images.size()}")
        # TODO!!!
        loss = model.compute_loss(generated_images, images, text_embeddings)
        total_loss += loss.item()
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # Update weights
            optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}")

    # Evaluation after each epoch
    val_loss = evaluate_model(model, val_loader, device)
    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}"
    )

model_save_path = "model_weight/"
final_save_path = os.path.join(model_save_path, "LoRAniDiff.pt")
torch.save(model.state_dict(), final_save_path)
