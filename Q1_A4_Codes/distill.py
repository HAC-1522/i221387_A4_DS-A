import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from common import seed_everything, get_device, get_loaders, ensure_dirs, RESULTS, write_json, environment_info, save_checkpoint, evaluate
from models import build_model, get_feature_vector, feature_dim

STUDENTS = ["mobilenetv2", "squeezenet", "efficientnetlite0"]

def kd_loss(student_logits, teacher_logits, targets, temperature=4.0, alpha=0.5):
    hard = F.cross_entropy(student_logits, targets)
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean"
    ) * (temperature ** 2)
    return alpha * soft + (1.0 - alpha) * hard, hard.detach(), soft.detach()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=STUDENTS, required=True)
    p.add_argument("--teacher", default="artifacts/teacher_resnet50.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5, help="Weight of soft-target loss")
    p.add_argument("--feature-weight", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    seed_everything(a.seed); ensure_dirs()
    device = get_device()
    train_loader, val_loader, _, _ = get_loaders(a.batch_size, a.num_workers)

    teacher = build_model("resnet50").to(device)
    payload = torch.load(a.teacher, map_location=device, weights_only=False)
    teacher.load_state_dict(payload["state_dict"])
    teacher.eval()
    for param in teacher.parameters(): param.requires_grad = False

    student = build_model(a.model).to(device)
    projector = nn.Linear(feature_dim(student, a.model), feature_dim(teacher, "resnet50")).to(device)
    optimizer = torch.optim.SGD(list(student.parameters()) + list(projector.parameters()),
                                lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    history, best_acc = [], -1.0
    start_epoch = 1
    latest_path = f"artifacts/{a.model}_distilled_latest.pt"

    if a.resume:
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"No recovery checkpoint found: {latest_path}")

        state = torch.load(latest_path, map_location=device, weights_only=False)
        student.load_state_dict(state["student_state_dict"])
        projector.load_state_dict(state["projector_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        history = state["history"]
        best_acc = state["best_accuracy"]
        start_epoch = state["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, a.epochs + 1):
        student.train(); projector.train()
        sums = {"total": 0.0, "hard": 0.0, "soft": 0.0, "feature": 0.0}
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f"KD {a.model} epoch {epoch}/{a.epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
                t_features = get_feature_vector(teacher, "resnet50", x)
            s_logits = student(x)
            s_features = get_feature_vector(student, a.model, x)
            aligned = projector(s_features)
            loss_kd, hard, soft = kd_loss(s_logits, t_logits, y, a.temperature, a.alpha)
            f_loss = F.mse_loss(F.normalize(aligned, dim=1), F.normalize(t_features, dim=1))
            loss = loss_kd + a.feature_weight * f_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            n = y.size(0)
            sums["total"] += loss.item() * n
            sums["hard"] += hard.item() * n
            sums["soft"] += soft.item() * n
            sums["feature"] += f_loss.item() * n
            correct += (s_logits.argmax(1) == y).sum().item(); total += n
            loop.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()
        metrics = evaluate(student, val_loader, device)
        row = {"epoch": epoch, "train_loss": sums["total"]/total,
               "hard_loss": sums["hard"]/total, "soft_loss": sums["soft"]/total,
               "feature_loss": sums["feature"]/total,
               "train_accuracy": 100.0*correct/total, **metrics}
        history.append(row); print(row)

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            save_checkpoint(f"artifacts/{a.model}_distilled.pt", student, a.model, epoch, metrics,
                            {"temperature": a.temperature, "alpha": a.alpha,
                             "feature_weight": a.feature_weight})

        write_json(RESULTS / f"{a.model}_distillation_history.json",
                   {"environment": environment_info(), "history": history,
                    "temperature": a.temperature, "alpha": a.alpha,
                    "feature_weight": a.feature_weight})

        torch.save({
            "epoch": epoch,
            "student_state_dict": student.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_accuracy": best_acc,
            "history": history
        }, latest_path)

        print(f"Recovery checkpoint saved after epoch {epoch}")
    write_json(RESULTS / f"{a.model}_distillation_history.json",
               {"environment": environment_info(), "history": history,
                "temperature": a.temperature, "alpha": a.alpha,
                "feature_weight": a.feature_weight})
if __name__ == "__main__": main()
