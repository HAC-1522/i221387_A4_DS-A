import argparse
from pathlib import Path
from common import seed_everything, get_device, get_loaders, ensure_dirs, RESULTS, environment_info, write_json
from models import build_model
from train_utils import train_classifier

STUDENTS = ["mobilenetv2", "squeezenet", "efficientnetlite0"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=STUDENTS, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    seed_everything(a.seed); ensure_dirs()
    train, val, _, _ = get_loaders(a.batch_size, a.num_workers)
    model = build_model(a.model)
    hist = train_classifier(model, train, val, get_device(), a.epochs, a.lr,
                            f"artifacts/{a.model}_baseline.pt", a.model)
    write_json(RESULTS / f"{a.model}_baseline_history.json",
               {"environment": environment_info(), "history": hist})
if __name__ == "__main__": main()
