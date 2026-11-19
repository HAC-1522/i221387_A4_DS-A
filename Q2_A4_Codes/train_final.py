import argparse, json
from pathlib import Path
import torch
from models import SearchableCNN
from data import build_loaders
from engine import fit, evaluate, get_device, seed_everything

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="artifacts/best_architecture.json")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default=".")
    args = p.parse_args()

    seed_everything(args.seed)
    with open(args.config) as f:
        architecture = json.load(f)
    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, architecture["input_resolution"], args.batch_size,
        args.num_workers, train_subset=45000, val_subset=5000, seed=args.seed
    )
    device = get_device(args.device)
    model = SearchableCNN(architecture)
    model, history, best_val = fit(model, train_loader, val_loader, args.epochs, device)
    test_loss, test_acc = evaluate(model, test_loader, device)

    out = Path(args.output_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": architecture,
        "model_state_dict": model.state_dict(),
        "best_validation_accuracy": best_val,
        "test_accuracy": test_acc,
        "history": history
    }, out / "artifacts/final_best_model.pt")
    with open(out / "artifacts/final_architecture.json", "w") as f:
        json.dump(architecture, f, indent=2)
    with open(out / "results/final_metrics.json", "w") as f:
        json.dump({"best_validation_accuracy": best_val, "test_loss": test_loss,
                   "test_accuracy": test_acc}, f, indent=2)
    print(f"Final test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
