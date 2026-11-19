import argparse, json, time
import numpy as np
import onnxruntime as ort
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="onnx/best_architecture.onnx")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--architecture", default="artifacts/final_architecture.json")
    args = p.parse_args()

    with open(args.architecture) as f:
        cfg = json.load(f)
    transform = transforms.Compose([
        transforms.Resize((cfg["input_resolution"], cfg["input_resolution"])),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_set = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    correct, total = 0, 0
    start = time.perf_counter()
    for images, labels in loader:
        logits = session.run(None, {input_name: images.numpy()})[0]
        pred = np.asarray(logits).argmax(axis=1)
        correct += int((pred == labels.numpy()).sum())
        total += labels.size(0)
    elapsed = time.perf_counter() - start
    print(f"ONNX Runtime CPU accuracy: {100*correct/total:.2f}%")
    print(f"Images: {total} | Total time: {elapsed:.3f}s | Throughput: {total/elapsed:.2f} images/s")

if __name__ == "__main__":
    main()
