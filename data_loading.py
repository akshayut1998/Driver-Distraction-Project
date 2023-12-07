# data_loading.py
from torch.utils.data import DataLoader, random_split,ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from Seed_LR import set_random_seeds

def load_split_data(directory, means, stds, batch_size=32, shuffle=True):
    set_random_seeds()
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds),
    ])

    # Load dataset
    dataset = ImageFolder(root=directory, transform=transform)

    # Calculate sizes of each split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size],generator=torch.Generator().manual_seed(42))

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    combined_loader = DataLoader(ConcatDataset([train_set, val_set]), batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader, combined_loader , train_set , val_set , test_set , dataset
