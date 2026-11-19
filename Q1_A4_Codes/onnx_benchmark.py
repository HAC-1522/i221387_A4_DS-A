
import time
import csv
from pathlib import Path
import numpy as np
import onnxruntime as ort

from common import get_test_loader

MODELS = {
    "MobileNetV2-Baseline": "onnx/mobilenetv2_baseline.onnx",
    "MobileNetV2-Distilled": "onnx/mobilenetv2_distilled.onnx",
    "SqueezeNet-Baseline": "onnx/squeezenet_baseline.onnx",
    "SqueezeNet-Distilled": "onnx/squeezenet_distilled.onnx",
    "EfficientNetLite0-Baseline": "onnx/efficientnetlite0_baseline.onnx",
    "EfficientNetLite0-Distilled": "onnx/efficientnetlite0_distilled.onnx",
}

def get_model_size(path):
    size = Path(path).stat().st_size
    external = Path(path + ".data")
    if external.exists():
        size += external.stat().st_size
    return size / (1024 * 1024)

def main():
    batch_size = 128
    warmup_batches = 10

    loader, _ = get_test_loader(batch_size, 2)
    rows = []

    for label, path in MODELS.items():
        print(f"\nBenchmarking: {label}")

        if not Path(path).exists():
            print("Skipping: ONNX file missing")
            continue

        session = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"]
        )

        input_name = session.get_inputs()[0].name
        correct = 0
        total = 0
        timed_images = 0
        elapsed = 0.0

        for batch_index, (images, targets) in enumerate(loader):
            x = images.numpy().astype(np.float32)
            y = targets.numpy()

            if batch_index < warmup_batches:
                logits = session.run(None, {input_name: x})[0]
            else:
                start = time.perf_counter()
                logits = session.run(None, {input_name: x})[0]
                elapsed += time.perf_counter() - start
                timed_images += len(y)

            correct += int((np.argmax(logits, axis=1) == y).sum())
            total += len(y)

        row = {
            "model": label,
            "test_accuracy_percent": round(100 * correct / total, 4),
            "mean_cpu_latency_ms_per_image": round(
                elapsed * 1000 / max(timed_images, 1), 6
            ),
            "onnx_model_size_mb": round(get_model_size(path), 4),
            "test_images_measured": total,
            "timed_images": timed_images
        }

        rows.append(row)
        print(row)

    output = Path("results/onnx_benchmark.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print("\nSaved:", output)

if __name__ == "__main__":
    main()
