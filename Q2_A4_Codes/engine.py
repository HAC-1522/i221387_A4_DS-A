import os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        loss_sum += loss.item() * labels.size(0)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)

def fit(model, train_loader, val_loader, epochs, device, lr=0.05, weight_decay=5e-4):
    model.to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9,
                    weight_decay=weight_decay, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_acc, best_state = -1.0, None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
            "val_loss": val_loss, "val_accuracy": val_acc
        })
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:02d}/{epochs} | train_acc={train_acc:.2f}% | val_acc={val_acc:.2f}%")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_acc
