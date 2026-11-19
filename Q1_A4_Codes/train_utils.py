import torch
from tqdm import tqdm
from common import evaluate, save_checkpoint

def train_classifier(model, train_loader, test_loader, device, epochs, lr, output_path, model_name):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            loop.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()
        metrics = evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / total,
               "train_accuracy": 100.0 * correct / total, **metrics}
        history.append(row)
        print(row)
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            save_checkpoint(output_path, model, model_name, epoch, metrics)
    return history
