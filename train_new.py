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
from ldm.ldm_new import LoRAniDiff
from torch.utils.data.dataset import random_split
import pandas as pd # for path fixing

WIDTH = 512
HEIGHT = 512
learning_rate = 1e-4
batch_size = 16
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
alpha = 0.5 
seed = 1337
cfg_scale = 7.5
pt_file = "model_weight/LoRAniDiff.pt"

# tokenizer = CLIPTokenizer("model_weight/vocab.json", merges_file="model_weight/merges.txt")
# model.load_state_dict(torch.load("model_weight/LoRAniDiff.pt", map_location=device))

tokenizer = CLIPTokenizer("model_weight/vocab.json", merges_file="model_weight/merges.txt")
pt_file = "model_weight/LoRAniDiff.pt"

model = LoRAniDiff(tokenizer=tokenizer, device=device, seed=seed, width=WIDTH, height=HEIGHT).to(device)

class TextCapsDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        self.metadata_df = pd.read_parquet(parquet_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #print("=======================")
        #print(self.metadata_df.keys())
        #print("=======================")

        img_path = self.metadata_df.iloc[idx]['Path']
        image = Image.open(img_path).convert('RGB')
        caption = self.metadata_df.iloc[idx]['Caption']
        #print(f'caption: {caption}')
        #print(f'image: {image.size}')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'caption': caption}

transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# * Fix paths
# Step 1: Read the Parquet file
df = pd.read_parquet('./image/image_data.parquet')

# Step 2: Update the 'Path' column
def update_path(old_path):
    return 'image/downloads32/' + old_path.split('/')[-1]

df['Path'] = df['Path'].apply(update_path)

# Step 3: Write the DataFrame back to a Parquet file
df.to_parquet('./image/image_data.parquet')


# Check
#print(df['Path']) 

dataset = TextCapsDataset(os.path.join(BASE_DIR, 'image/image_data.parquet'), transform=transform) # prev: 'data/textcaps/metadata.parquet'
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

    #print(dataset.metadata_df['Name']) 

    for i, batch in tqdm(enumerate(train_loader)):
        #print("printing all the batch keys....")
        #print(batch.keys())

        
        #print(batch['image'])
        images = batch['image'].to(device)
        captions = batch['caption']

        #print(captions)
        # * Trim the captions to just be assistant output
        # Iterate through all the captions and only keep the text after the substring ASSISTANT: 
        for i in range(len(captions)):
            caption = captions[i]
            if "ASSISTANT:" in caption:
                caption = caption[caption.index("ASSISTANT:") + len("ASSISTANT:"):]
            captions[i] = caption
            # if the number of character in captions[i] is less than 77, pad it with spaces to make it 77 characters long
            #if len(captions[i]) < 77:
            #    captions[i] = captions[i] + " " * (77 - len(captions[i]))
            if len(captions[i]) > 77:
                captions[i] = captions[i][:77]

        
        # Should be fixed
        #print(captions)

        uncond_prompt = [""] * len(captions)

        #print(uncond_prompt)

        output_conditioned, output_unconditioned, text_embeddings = model.forward(captions, images, uncond_prompt)

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