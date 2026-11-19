import argparse, time, os, csv, torch
from pathlib import Path
from common import get_test_loader, model_size_mb
from models import build_model

CHECKPOINTS = {
    "MobileNetV2-Baseline": ("mobilenetv2","artifacts/mobilenetv2_baseline.pt"),
    "SqueezeNet-Baseline": ("squeezenet","artifacts/squeezenet_baseline.pt"),
    "EfficientNetLite0-Baseline": ("efficientnetlite0","artifacts/efficientnetlite0_baseline.pt"),
    "MobileNetV2-Distilled": ("mobilenetv2","artifacts/mobilenetv2_distilled.pt"),
    "SqueezeNet-Distilled": ("squeezenet","artifacts/squeezenet_distilled.pt"),
    "EfficientNetLite0-Distilled": ("efficientnetlite0","artifacts/efficientnetlite0_distilled.pt"),
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--output",default="results/student_benchmark.csv")
    a=p.parse_args()
    loader, _ = get_test_loader(a.batch_size, 2)
    device=torch.device("cpu") # deployment comparison is CPU-based
    rows=[]
    for label,(arch,path) in CHECKPOINTS.items():
        if not Path(path).exists():
            print(f"Skipping {label}: missing {path}"); continue
        model=build_model(arch)
        payload=torch.load(path,map_location="cpu",weights_only=False)
        model.load_state_dict(payload["state_dict"]); model.eval()
        correct,total=0,0; elapsed=0.0; batches=0; measured_images=0
        with torch.inference_mode():
            for bi,(x,y) in enumerate(loader):
                if bi < a.warmup:
                    logits = model(x)  # warm-up contributes to accuracy, not latency
                else:
                    start=time.perf_counter(); logits=model(x); elapsed += time.perf_counter()-start
                    batches += 1; measured_images += y.size(0)
                correct += (logits.argmax(1)==y).sum().item(); total += y.size(0)
        latency_ms=elapsed*1000/max(measured_images,1)
        rows.append({"model":label,"test_accuracy_percent":100*correct/max(total,1),
                     "mean_cpu_latency_ms_per_image":latency_ms,
                     "checkpoint_size_mb":model_size_mb(path),"test_images_measured":total})
        print(rows[-1])
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with open(a.output,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print(f"Saved {a.output}")
if __name__=="__main__": main()
