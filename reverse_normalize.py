import torchvision.transforms as transforms

def reverse_normalize(tensor, mean, std):
    # Undo normalization
    reverse_mean = [-m / s for m, s in zip(mean, std)]
    reverse_std = [1 / s for s in std]
    reverse_transform = transforms.Normalize(mean=reverse_mean, std=reverse_std)

    # Convert to a PyTorch tensor
    original_tensor = reverse_transform(tensor)

    return original_tensor