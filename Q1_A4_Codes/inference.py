import argparse, torch
from PIL import Image
from torchvision import transforms
from common import get_device
from models import build_model

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
        choices=["mobilenetv2_baseline","squeezenet_baseline","efficientnetlite0_baseline",
                 "mobilenetv2_distilled","squeezenet_distilled","efficientnetlite0_distilled"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    a = p.parse_args()
    device = get_device()
    architecture = a.model.replace("_baseline","").replace("_distilled","")
    model = build_model(architecture)
    payload = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"]); model.to(device).eval()
    tf = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    x = tf(Image.open(a.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = model(x).softmax(1)[0]
    values, indices = probs.topk(5)
    print("Top-5 predictions:")
    for score, idx in zip(values.tolist(), indices.tolist()):
        print(f"class_index={idx:02d} confidence={score:.4f}")
if __name__ == "__main__": main()
