"""
Script for training a model on specific datasets with optional Weights & Biases logging.
Supports training on 'pixiv' and 'textcaps' datasets, including functionality for model evaluation,
batch processing, and optional image resizing.
"""

import argparse
import datetime
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTokenizer
import wandb
from ldm.ldm import LoRAniDiff
from ldm.dataset import ImageCaptionDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the LoRAniDiff model on specified dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pixiv", "textcaps"],
        required=True,
        help="Dataset to train on, either 'pixiv' or 'textcaps'.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight between reconstruction and CLIP-guided losses.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use WandB for logging. Defaults to False.",
    )
    return parser.parse_args()


def setup_dataset(args, BASE_DIR):
    """Set up the dataset for training and validation."""
    if args.dataset == "textcaps":
        parquet_path = os.path.join(BASE_DIR, "data/textcaps/metadata.parquet")
        images_path = os.path.join(BASE_DIR, "data/textcaps/processed")
    elif args.dataset == "pixiv":
        parquet_path = os.path.join(BASE_DIR, "data/pixiv/pixiv_image_caption.parquet")
        images_path = os.path.join(BASE_DIR, "data/pixiv/pixiv_images")
    else:
        raise ValueError("Invalid dataset name. Please specify 'textcaps' or 'pixiv'.")

    transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)), transforms.ToTensor()]
    )

    dataset = ImageCaptionDataset(parquet_path, images_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader


def setup_model(args, BASE_DIR, device):
    """Initialize and return the model."""
    tokenizer = CLIPTokenizer(
        "model_weight/vocab.json", merges_file="model_weight/merges.txt"
    )
    # MODEL_FILE DEPRECATING
    # MODEL_FILE = os.path.join(BASE_DIR, "model_weight/v1-5-pruned-emaonly.ckpt")
    PT_FILE = os.path.join(BASE_DIR, "model_weight/LoRAniDiff.pt")

    model = LoRAniDiff(
        tokenizer=tokenizer, device=device, width=args.width, height=args.height
    ).to(device)
    model.load_state_dict(torch.load(PT_FILE, map_location=device))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    return model


def evaluate_model(model, data_loader, device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, captions in data_loader:
            images = images.to(device)
            captions = captions.to(device)
            val_loss = model(
                images, captions
            )  # Assuming the model's forward method returns a loss
            total_val_loss += val_loss.item()
    return total_val_loss / len(data_loader)


def main():
    args = parse_arguments()

    if args.wandb:
        wandb.init(project="your_project_name", config=vars(args))
        wandb.config.update(args)
        wandb.run.save()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    train_loader, val_loader = setup_dataset(args, BASE_DIR)
    model = setup_model(args, BASE_DIR, device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("Training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            images = batch["image"].to(device)
            captions = batch["caption"]
            uncond_prompt = [""] * len(captions)
            # HENRY TODO
            generated_images, context, uncond_context = model(
                prompt=captions, uncond_prompt=uncond_prompt, images=images
            )
            loss = model.compute_loss(generated_images, images, context)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Training Loss: {avg_loss:.4f}")
        if args.wandb:
            wandb.log({"training_loss": avg_loss, "epoch": epoch})

        val_loss = evaluate_model(model, val_loader, device)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {val_loss:.4f}")
        if args.wandb:
            wandb.log({"validation_loss": val_loss, "epoch": epoch})

    # Save the trained model
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Construct the file name with date, time, and epochs
    OUTPUT_FILE = f"output_weight/{current_time}_epochs_{args.epochs}.pt"
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, OUTPUT_FILE)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
