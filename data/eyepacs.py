import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import zipfile
import torch

# Custom dataset class for Diabetic Retinopathy Detection
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.data.columns = ['image', 'label']  # Ensure the CSV has correct column names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0] + '.jpeg')
        image = Image.open(img_name)
        label = int(self.data.iloc[idx, 1])  # Ensure label is an integer

        if self.transform:
            image = self.transform(image)

        return image, label

# DataLoader function
def get_dataloaders(data_dir, csv_file, batch_size=32, num_workers=4):
    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    dataset = DiabeticRetinopathyDataset(csv_file=csv_file, root_dir=os.path.join(data_dir, 'train'), transform=train_transform)

    # Split into train, validation, and test datasets
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(valid_dataset))
    print("Test dataset size:", len(test_dataset))
    
    return train_loader, valid_loader, test_loader
