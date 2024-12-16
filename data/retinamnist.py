import medmnist
from medmnist import RetinaMNIST, INFO
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size: int = 32, download: bool = True, num_workers: int = 4):
    """
    Returns a DataLoader for the Blood MNIST dataset with specified split, batch size, shuffle, and download options.
    
    Parameters:
    - split (str): Dataset split to load ('train', 'val', or 'test')
    - batch_size (int): Size of data batches (default: 32)
    - shuffle (bool): Whether to shuffle the data (default: True for train, False for val/test)
    - download (bool): Whether to download the dataset if not already downloaded (default: True)
    
    Returns:
    - DataLoader: DataLoader for the specified split
    """

    # Define data transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize dataset with selected transforms
    train_dataset = RetinaMNIST(root='data', split='train', transform=train_transforms, download=download, size=224)
    valid_dataset = RetinaMNIST(root='data', split='val', transform=test_transforms, download=download, size=224)
    test_dataset = RetinaMNIST(root='data', split='test', transform=test_transforms, download=download, size=224)
    
    x, y = train_dataset[0]

    print(x.shape, y.shape)
    
    # Initialize DataLoader with selected dataset and options
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(valid_dataset))
    print("Test dataset size:", len(test_dataset))
    
    return train_loader, valid_loader, test_loader
    
    
    




