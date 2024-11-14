import medmnist
from medmnist import PathMNIST, INFO
from torch.utils.data import DataLoader
from torchvision import transforms

def get_dataloader(batch_size: int = 32, download: bool = True, num_workers: int = 4):
    """
    Returns DataLoaders for the PathMNIST dataset with 224x224 image size for RGB images.
    """
    data_flag = 'pathmnist'
    
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
    
    # Define data transforms for 224x224 resolution, suitable for RGB images
    train_transforms = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize images to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # RGB normalization
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load datasets with specified transforms
    train_dataset = PathMNIST(root='data', split='train', transform=train_transforms, download=download)
    valid_dataset = PathMNIST(root='data', split='val', transform=test_transforms, download=download)
    test_dataset = PathMNIST(root='data', split='test', transform=test_transforms, download=download)
    
    # print(train_dataset)
    # print(valid_dataset)
    # print(test_dataset)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader