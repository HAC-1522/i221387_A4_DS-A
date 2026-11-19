import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

def build_loaders(data_dir, resolution, batch_size=128, num_workers=2,
                  train_subset=12000, val_subset=3000, seed=42):
    transform_train = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_aug = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    train_eval = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_eval)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_eval)

    total = len(train_aug)
    n_train = min(train_subset, total - 1) if train_subset else total - 5000
    n_val = min(val_subset, total - n_train) if val_subset else total - n_train
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    train_idx, val_idx = indices[:n_train], indices[n_train:n_train+n_val]
    train_set = Subset(train_aug, train_idx)
    val_set = Subset(train_eval, val_idx)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_set, shuffle=True, **kwargs),
        DataLoader(val_set, shuffle=False, **kwargs),
        DataLoader(test_set, shuffle=False, **kwargs),
    )
