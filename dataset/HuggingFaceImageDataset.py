import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from dataset.imagenet_dataset import get_train_transforms
from transformers import CLIPTokenizer, SiglipImageProcessor  # Added SiglipImageProcessor

class HuggingFaceImageDataset(Dataset):
    def __init__(self, dataset_name, split='train', transform=None,processor = None):
        """
        Args:
            dataset_name (str): The name of the dataset on Hugging Face Hub (e.g., 'username/dataset_name').
            split (str): Which split of the dataset to load ('train', 'test', etc.).
            transform (callable, optional): A function/transform to apply to the images.
        """
        # Load the dataset from the Hugging Face Hub
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transform
        self.processor = processor
        print(f"loaded huggingface dataset {dataset_name} split {split}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load the image and text for the given index
        item = self.dataset[idx]
        image = item['image']  # this is a PIL Image in HF dataset by default
        #prompt = item['text']  # this is the text prompt associated with the image

        # Apply transformations if any
        if self.transform:
            image_transform = self.transform(image)
        if self.processor is not None:
            images_siglip = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            return image_transform,images_siglip

        return image_transform

def get_train_val_datasets(dataset_name = "AmitIsraeli/pops_10k",transform_2 = False ,train_ratio=0.9, batch_size=32,final_reso = 256,siglip_model_name ="google/siglip-base-patch16-224",hugging_face_token = None ):
    mid_reso = round(1.125 * final_reso)  # mid_reso = 252
    transform = get_train_transforms(final_reso, mid_reso, hflip = False)
    if transform_2:
        processor = SiglipImageProcessor.from_pretrained(siglip_model_name,token=hugging_face_token)
    else:
        processor = None
    dataset = HuggingFaceImageDataset(dataset_name, transform=transform,processor=processor)

    # Calculate lengths for train and validation splits
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader