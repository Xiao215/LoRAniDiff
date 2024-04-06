import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader,Dataset
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
learning_rate = 1e-4
batch_size = 16
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
alpha = 0.5 
seed = 1337
cfg_scale = 7.5
tokenizer = CLIPTokenizer("model_weight/vocab.json", merges_file="model_weight/merges.txt")
model = LoRAniDiff(tokenizer=tokenizer, device=device, seed=seed, width=WIDTH, height=HEIGHT).to(device)
model.load_state_dict(torch.load("model_weight/LoRAniDiff_final_model.pt", map_location=device))

class TextCapsDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        self.metadata_df = pd.read_parquet(parquet_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.metadata_df.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        caption = self.metadata_df.iloc[idx]['caption']
        print(f'caption: {caption}')
        print(f'image: {image.size}')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'caption': caption}

transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset = TextCapsDataset(os.path.join(BASE_DIR, 'data/textcaps/metadata.parquet'), transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

optimizer = Adam(model.parameters(), lr=learning_rate)

def evaluate_model(model, data_loader, device):
    model.eval() 
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            captions = batch['caption']
            generated_images, text_embeddings = model.generate(captions, images)
            loss = model.compute_loss(generated_images, images, text_embeddings)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

print("Training model...")
for epoch in range(epochs):
    model.train() 
    total_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        images = batch['image'].to(device)
        captions = batch['caption']

        uncond_prompt = [""] * len(captions)
        output_conditioned, output_unconditioned, text_embeddings = model.forward(images, captions, uncond_prompt, tokenizer)

        cfg_output = cfg_scale * (output_conditioned - output_unconditioned) + output_unconditioned

        loss = model.compute_loss(cfg_output, images, text_embeddings)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}')

    val_loss = evaluate_model(model, val_loader, device)
    print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}')

model_path = os.path.join("model_weight/", 'LoRAniDiff_trained.pt')
torch.save(model.state_dict(), model_path)