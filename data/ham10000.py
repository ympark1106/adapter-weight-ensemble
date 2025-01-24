import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Custom dataset class
class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_mapping = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
        self.data['encoded_labels'] = self.data.iloc[:, 1:].idxmax(axis=1).map(self.class_mapping)
        
        # print("Encoded labels:", self.data['encoded_labels'].unique())
        # print("Label counts:", self.data['encoded_labels'].value_counts())


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        # label = self.data.iloc[idx, 1]
        label = self.data.iloc[idx]['encoded_labels']
        

        if self.transform:
            image = self.transform(image)

        return image, label

# DataLoader function
def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.ColorJitter(),
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
    
    train_csv = data_dir + 'ISIC2018_Task3_Training_GroundTruth.csv'
    valid_csv = data_dir + 'ISIC2018_Task3_Validation_GroundTruth.csv'
    test_csv = data_dir + 'ISIC2018_Task3_Test_GroundTruth.csv'
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'
    
    # Datasets
    train_dataset = HAM10000Dataset(csv_file=train_csv, root_dir=train_dir, transform=train_transform)
    valid_dataset = HAM10000Dataset(csv_file=valid_csv, root_dir=valid_dir, transform=test_transform)
    test_dataset = HAM10000Dataset(csv_file=test_csv, root_dir=test_dir, transform=test_transform)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(valid_dataset))
    print("Test dataset size:", len(test_dataset))
    
    return train_loader, valid_loader, test_loader
