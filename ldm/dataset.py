"""
This module defines a PyTorch Dataset for an image captioning task. It loads image-caption pairs
from a directory of images and a Parquet file containing the metadata. Optionally, transformations
can be applied to each image.
"""

import os
from typing import Callable

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class ImageCaptionDataset(Dataset):
    """Dataset class for an image captioning dataset."""

    def __init__(
        self, parquet_file: str, image_dir: str, transform: Callable | None = None
    ):
        """
        Initializes the dataset.

        Args:
            parquet_file (str): Path to the parquet file with metadata.
            image_dir (str): Directory with all the images.
            transform (Callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata_df = pd.read_parquet(parquet_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> dict:
        """Fetches the image-caption pair at the specified index in the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.metadata_df.iloc[idx]["image_name"]  # Fetch the image name
        img_path = os.path.join(self.image_dir, img_name)  # Construct the image path
        image = Image.open(img_path).convert("RGB")  # Load and convert the image
        caption = self.metadata_df.iloc[idx]["caption"]  # Fetch the associated caption

        if self.transform:  # Apply transformation, if any
            image = self.transform(image)

        return {"image": image, "caption": caption}
