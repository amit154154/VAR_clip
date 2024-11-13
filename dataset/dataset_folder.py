import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from dataset.imagenet_dataset import get_train_transforms

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff'))]
        self.transform = transform if transform else transforms.ToTensor()  # Default to tensor transform if none provided

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Convert image to RGB

        if self.transform:
            image = self.transform(image)

        return image

def get_train_val_datasets(folder_path, train_ratio=0.8, batch_size=32,final_reso = 256):
    mid_reso = round(1.125 * final_reso)  # mid_reso = 252
    transform = get_train_transforms(final_reso, mid_reso, hflip = False)
    dataset = ImageFolderDataset(folder_path, transform=transform)

    # Calculate lengths for train and validation splits
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader