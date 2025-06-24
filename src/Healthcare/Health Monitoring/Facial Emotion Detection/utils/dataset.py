import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# Set up transformations
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(48, padding=4),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

def get_dataloaders(data_dir="data/data", batch_size=64):
    """
    Returns train and test dataloaders given a dataset directory.
    Expects the following structure:
        data/
          └── train/
              ├── Angry/
              ├── Happy/
              └── ...
          └── test/
              ├── Angry/
              ├── Happy/
              └── ...
    """

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, train_dataset.classes
