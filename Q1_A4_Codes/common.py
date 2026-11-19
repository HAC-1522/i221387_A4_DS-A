import os, json, random, time, platform
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path("data")
ARTIFACTS = Path("artifacts")
RESULTS = Path("results")
NUM_CLASSES = 100
CLASS_NAMES = None

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders(batch_size=128, num_workers=2, data_dir=DATA_DIR, val_size=5000):
    # CIFAR-100 standard channel statistics
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # Keep the official test set untouched until final benchmarking.
    train_aug = datasets.CIFAR100(str(data_dir), train=True, download=True, transform=train_tf)
    train_eval = datasets.CIFAR100(str(data_dir), train=True, download=True, transform=eval_tf)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_aug), generator=generator).tolist()
    val_indices, train_indices = indices[:val_size], indices[val_size:]
    train_ds = torch.utils.data.Subset(train_aug, train_indices)
    val_ds = torch.utils.data.Subset(train_eval, val_indices)
    test_ds = datasets.CIFAR100(str(data_dir), train=False, download=True, transform=eval_tf)
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, test_ds.classes

def get_test_loader(batch_size=128, num_workers=2, data_dir=DATA_DIR):
    _, _, test_loader, classes = get_loaders(batch_size, num_workers, data_dir)
    return test_loader, classes

def ensure_dirs():
    ARTIFACTS.mkdir(exist_ok=True, parents=True)
    RESULTS.mkdir(exist_ok=True, parents=True)

def save_checkpoint(path, model, model_name, epoch=None, metrics=None, extra=None):
    payload = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    torch.save(payload, path)

def load_checkpoint(path, model, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return model, payload

def accuracy_from_logits(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item() * 100.0

def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"loss": loss_sum / max(total, 1), "accuracy": 100.0 * correct / max(total, 1)}

def model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def environment_info():
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "platform": platform.platform(),
    }
