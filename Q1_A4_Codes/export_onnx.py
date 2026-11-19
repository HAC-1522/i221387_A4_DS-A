import argparse, torch
from models import build_model

MODELS = ["mobilenetv2_baseline","squeezenet_baseline","efficientnetlite0_baseline",
          "mobilenetv2_distilled","squeezenet_distilled","efficientnetlite0_distilled"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    architecture = a.model.replace("_baseline","").replace("_distilled","")
    model = build_model(architecture)
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"]); model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy, a.output, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, do_constant_folding=True)
    print(f"Exported {a.model} to {a.output}")
if __name__ == "__main__": main()
