import argparse, json
from common import seed_everything, get_device, get_loaders, ensure_dirs, RESULTS, environment_info, write_json
from models import build_model
from train_utils import train_classifier

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    seed_everything(a.seed); ensure_dirs()
    train, val, _, _ = get_loaders(a.batch_size, a.num_workers)
    model = build_model("resnet50")
    hist = train_classifier(model, train, val, get_device(), a.epochs, a.lr,
                            "artifacts/teacher_resnet50.pt", "resnet50")
    write_json(RESULTS / "teacher_training_history.json",
               {"environment": environment_info(), "history": hist})
if __name__ == "__main__": main()
