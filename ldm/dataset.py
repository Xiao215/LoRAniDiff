from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import os
from typing import Callable


class ImageCaptionDataset(Dataset):
    """Dataset class for an image captioning dataset."""

    def __init__(
        self, parquet_file: str, image_dir: str, transform: Callable | None = None
    ):
        """
        Args:
            parquet_file (string): Path to the parquet file with metadata.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata_df = pd.read_parquet(parquet_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Fetch the image name from the dataframe
        img_name = self.metadata_df.iloc[idx]["image_name"]
        # Construct the full image path
        img_path = os.path.join(self.image_dir, img_name)
        # Load the image
        image = Image.open(img_path).convert("RGB")
        # Fetch the caption associated with the image
        caption = self.metadata_df.iloc[idx]["caption"]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "caption": caption}
