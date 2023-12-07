def set_random_seeds(seed_value=42):
    import random
    import numpy as np
    import torch

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def find_lr(train_loader, model, criterion, optimizer, device):
    from torch_lr_finder import LRFinder

    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
    lr_finder.plot()
    lr_finder.reset()

    # Get the learning rate values
    lrs = lr_finder.history['lr']
    losses = lr_finder.history['loss']

    # Find the learning rate value where the loss is minimum
    lr_min_loss = lrs[losses.index(min(losses))]

    return lr_min_loss