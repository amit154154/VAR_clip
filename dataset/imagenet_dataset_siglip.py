import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import CLIPTokenizer, SiglipImageProcessor  # Added SiglipImageProcessor
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


from torch.utils.data import Dataset
import torch

from torch.utils.data import Dataset, Subset
import torch


class ImagenetDataset(Dataset):
    def __init__(self, dataset, transform=None, processor=None, tokenizer=None, max_length=64, only_class=None,
                 max_pre_filter_samples=10000):
        """
        Initializes the ImagenetDataset.

        Args:
            dataset (Dataset): The original dataset to wrap.
            transform (callable, optional): A function/transform to apply to the images.
            processor (callable, optional): Processor for SiglipImageProcessor.
            tokenizer (callable, optional): Tokenizer for captions.
            max_length (int, optional): Maximum length for tokenization.
            only_class (list of int, optional): List of class labels to include. If None, include all classes.
            max_pre_filter_samples (int, optional): Maximum number of samples to consider for pre-filtering.
                                                  Only relevant if `only_class` is specified.
        """
        self.transform = transform
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        if only_class is not None:
            # Determine the number of samples to consider for filtering
            num_samples_to_filter = min(max_pre_filter_samples, len(dataset))

            print(f"Pre-filtering the first {num_samples_to_filter} samples for classes: {only_class}")

            # Pre-filter the dataset to include only samples with labels in only_class within the first `num_samples_to_filter` samples
            filtered_indices = [
                idx for idx in range(num_samples_to_filter)
                if dataset[idx]['label'] in only_class
            ]

            if not filtered_indices:
                raise ValueError("No samples found for the specified `only_class` within the first "
                                 f"{num_samples_to_filter} samples.")

            self.dataset = Subset(dataset, filtered_indices)
            print(f"Number of samples after filtering: {len(self.dataset)}")
        else:
            # Optionally, limit the dataset to the first `max_pre_filter_samples` samples even if `only_class` is None
            # Uncomment the following lines if you want to limit the dataset size regardless of `only_class`
            # num_samples = min(max_pre_filter_samples, len(dataset))
            # self.dataset = Subset(dataset, list(range(num_samples)))
            # print(f"Dataset limited to the first {num_samples} samples.")

            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            sample = self.dataset[idx]
            image = sample['image']
            label = sample['label']
            caption = sample.get('caption_enriched', "")

            # Ensure the image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Transform the main image
            if self.transform:
                images = self.transform(image)
            else:
                images = image

            # Process images_siglip using SiglipImageProcessor
            if self.processor:
                images_siglip = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                images_siglip = None

            # Tokenize the caption for description
            if self.tokenizer:
                caption_encoding = self.tokenizer(
                    caption,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                description = {
                    'input_ids': caption_encoding['input_ids'].squeeze(0),
                    'attention_mask': caption_encoding['attention_mask'].squeeze(0)
                }
            else:
                description = None

            return {
                'images': images,
                'images_siglip': images_siglip,
                'label': label,
                'description': description
            }
        except Exception as e:
            print(f"Error processing sample at index {idx}: {e}")
            return None

def normalize_01_into_pm1(x):
    """
    Normalize tensor from [0, 1] to [-1, 1].

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Normalized tensor.
    """
    return x.mul(2.0).sub(1.0)


def print_aug(transform, label):
    """
    Print the transformations for debugging purposes.

    Args:
        transform (callable): Transformation pipeline.
        label (str): Label for the transformation.
    """
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')


def get_train_transforms(final_reso = 256, mid_reso = 288, hflip=False):
    """
    Define the transformation pipeline for training and validation.

    Args:
        final_reso (int): Final resolution after center cropping.
        mid_reso (int): Intermediate resolution for resizing.
        hflip (bool, optional): Whether to apply horizontal flipping.

    Returns:
        Compose: Composed transformations.
    """
    return transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.RandomHorizontalFlip() if hflip else transforms.Lambda(lambda x: x),
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        transforms.Lambda(normalize_01_into_pm1),
    ])


def dataset_collate_fn(batch):
    """
    Custom collate function to handle batches with 'images_siglip'.

    Args:
        batch (list): List of samples.

    Returns:
        dict: Batched samples.
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]

    images = torch.stack([item['images'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])

    # Handle captions (description)
    captions = batch[0]['description']
    if captions is not None:
        input_ids = torch.stack([item['description']['input_ids'] for item in batch])
        attention_mask = torch.stack([item['description']['attention_mask'] for item in batch])
        descriptions = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    else:
        descriptions = None

    # Handle images_siglip
    images_siglip = torch.stack([item['images_siglip'] for item in batch])

    return {
        'images': images,
        'images_siglip': images_siglip,
        'labels': labels,
        'description': descriptions
    }


def get_train_val_datasets(
        model_name="openai/clip-vit-base-patch32",
        siglip_model_name="facebook/siglip-vision-base",  # Specify the SigLIP model
        hugging_face_token = None,
        final_reso=256,
        do_hflip=False,
        batch_size=32,
        only_class= None
):
    """
    Prepare training and validation DataLoaders.

    Args:
        model_name (str, optional): Name of the CLIP tokenizer model.
        siglip_model_name (str, optional): Name of the SigLIP processor.
        final_reso (int, optional): Final resolution after cropping.
        do_hflip (bool, optional): Whether to apply horizontal flipping.
        batch_size (int, optional): Batch size for DataLoader.

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """

    # Initialize tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    # Initialize SigLIP processor
    processor = SiglipImageProcessor.from_pretrained(siglip_model_name,token=hugging_face_token)

    mid_reso = round(1.125 * final_reso)  # mid_reso = 256 * 1.125 = 288

    train_aug = get_train_transforms(final_reso, mid_reso, do_hflip)  # Define the transformations for training
    val_aug = get_train_transforms(final_reso, mid_reso, hflip=False)  # Define the transformations for validation

    # Optional: Print the transformations
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')

    # Load the streaming datasets
    train_dataset = load_dataset(
        "visual-layer/imagenet-1k-vl-enriched",
        split="train",
        streaming=False
    )
    val_dataset = load_dataset(
        "visual-layer/imagenet-1k-vl-enriched",
        split="validation",
        streaming=False
    )

    # Create the custom datasets
    train_streaming_dataset = ImagenetDataset(
        dataset=train_dataset,
        transform=train_aug,
        processor=processor,
        tokenizer=tokenizer,
        only_class = only_class
    )
    val_streaming_dataset = ImagenetDataset(
        dataset=val_dataset,
        transform=val_aug,
        processor=processor,
        tokenizer=tokenizer,
        only_class=only_class

    )

    # Create the DataLoaders
    train_dataloader = DataLoader(
        train_streaming_dataset,
        batch_size=batch_size,
        collate_fn=dataset_collate_fn
    )

    val_dataloader = DataLoader(
        val_streaming_dataset,
        batch_size=batch_size,
        collate_fn=dataset_collate_fn
    )

    return train_dataloader, val_dataloader