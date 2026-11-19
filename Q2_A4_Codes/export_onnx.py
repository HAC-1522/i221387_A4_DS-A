import argparse, json
from pathlib import Path
import torch
from models import SearchableCNN

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/final_best_model.pt")
    p.add_argument("--output", default="onnx/best_architecture.onnx")
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = SearchableCNN(architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    size = architecture["input_resolution"]
    dummy = torch.randn(1, 3, size, size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(output), input_names=["image"], output_names=["logits"],
        opset_version=17, dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False
    )
    import onnx
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    print("ONNX export and checker validation passed:", output)

if __name__ == "__main__":
    main()
