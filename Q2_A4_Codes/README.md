# Assignment 4 — Question 2: Hardware-Aware Neural Architecture Search

This project implements an evolutionary Neural Architecture Search (NAS) framework for lightweight CNNs on CIFAR-10. It generates candidate architectures, trains and validates them, measures real CPU inference latency and serialized model size, records process memory where supported, rejects candidates outside configurable constraints, retrains the selected architecture from scratch, and exports it to ONNX for ONNX Runtime CPU inference.

## Search space

Each candidate varies:
- CNN depth: 2–6 blocks
- Channels per block: 16, 24, 32, 48, 64, 96, 128
- Kernel sizes: 3 or 5
- Convolution: standard or depthwise-separable
- Activation: ReLU, ReLU6, GELU, SiLU
- CIFAR-10 input resolution: 24, 28, 32, 40

## Important measurement details

- Latency is measured on CPU with batch size 1, using `time.perf_counter`; warm-up iterations are excluded. Median and mean are recorded.
- Model storage is measured from the actual serialized PyTorch state-dictionary file.
- Memory records the current process RSS using `psutil` when available. This is a process-level snapshot, not an isolated per-model peak allocation. The report must describe it accurately.
- Default constraints (configurable): 10 ms median CPU latency, 15 MB serialized model size, 512 MB process RSS.
- These limits are starting deployment targets, not universal edge-device standards. Adjust them to match the target hardware or instructor's specified limits.
- Candidate search uses a training subset and short training budget for cost control. The final model is trained from scratch with 45,000 training and 5,000 validation images. The official CIFAR-10 test set is reserved for final evaluation.

## Setup

```bash
pip install -r requirements.txt
```

For Kaggle, enable GPU for training. CPU latency measurement remains explicitly on CPU.

## Run

```bash
python main.py --population-size 8 --generations 5 --candidate-epochs 3
```

The search writes:
- `results/search_history.csv`
- `artifacts/best_architecture.json`
- `artifacts/best_search_record.json`
- `search_config.json`

If no candidate satisfies every constraint, the framework saves the highest-validation-accuracy observed candidate with `feasible=false` and a warning. Do not present it as constraint-compliant.

Retrain the selected architecture from scratch:

```bash
python train_final.py --epochs 30
```

Export to ONNX:

```bash
python export_onnx.py
```

Run ONNX Runtime CPU inference:

```bash
python onnx_inference.py
```

## Reproducibility and reporting

Record the Kaggle GPU/CPU, Python and PyTorch versions, search settings, constraint values, number of candidates evaluated, rejected candidates and reasons, selected architecture, final validation/test accuracy, measured CPU latency, model size, process RSS caveat, ONNX validation, and ONNX Runtime accuracy.

Do not fabricate results. All reported measurements must come from the executed pipeline.
