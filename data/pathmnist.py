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
    
    # Load datasets with specified transforms
    train_dataset = PathMNIST(root='data', split='train', transform=train_transforms, download=download, size=224)
    valid_dataset = PathMNIST(root='data', split='val', transform=test_transforms, download=download, size=224)
    test_dataset = PathMNIST(root='data', split='test', transform=test_transforms, download=download, size=224)
    
    # print(train_dataset)
    # print(valid_dataset)
    # print(test_dataset)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader