import torch
from torch.utils.data import IterableDataset, DataLoader,Dataset

from datasets import load_dataset
from transformers import CLIPTokenizer
from torchvision import transforms
from torchvision.transforms import InterpolationMode



class ImagenetDataset(Dataset):  # Change IterableDataset to Dataset
    def __init__(self, dataset, transform=None, tokenizer=None, max_length=64):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        try:
            image = sample['image']
            label = sample['label']
            caption = sample['caption_enriched']

            # Transform the image
            if self.transform:
                image = self.transform(image)

            # Tokenize the caption
            if self.tokenizer:
                caption_encoding = self.tokenizer(
                    caption,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
            else:
                caption_encoding = None

            return {
                'image': image,
                'label': label,
                'caption': caption_encoding
            }
        except Exception as e:
            print(f"Error processing sample {sample.get('image_id', 'unknown')}: {e}")
            return None  # Optionally handle errors differently

def normalize_01_into_pm1(x):
    # Normalize x from [0, 1] to [-1, 1] by (x * 2) - 1
    return x.mul(2.0).sub(1.0)


# Function to print transformations (optional)
def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')

def get_train_transforms(final_reso, mid_reso, hflip= False):
    return transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.RandomHorizontalFlip() if hflip else transforms.Lambda(lambda x: x),
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        transforms.Lambda(normalize_01_into_pm1),
    ])

def dataset_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    input_ids = torch.stack([item['caption']['input_ids'].squeeze(0) for item in batch])
    attention_mask = torch.stack([item['caption']['attention_mask'].squeeze(0) for item in batch])

    captions = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    return {
        'images': images,
        'labels': labels,
        'captions': captions
    }

def get_train_val_datasets(
        model_name = "openai/clip-vit-base-patch32",
        final_reso = "256",
        do_hflip = False,
        batch_size = 32):

    # Initialize tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    mid_reso = round(1.125 * final_reso)  # mid_reso = 252

    train_aug = get_train_transforms(final_reso, mid_reso, do_hflip)    # Define the transformations for training
    val_aug = get_train_transforms(final_reso, mid_reso, hflip = False)     # Define the transformations for validation


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
        tokenizer=tokenizer
    )
    val_streaming_dataset = ImagenetDataset(
        dataset=val_dataset,
        transform=val_aug,
        tokenizer=tokenizer
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

    return train_dataloader,val_dataloader