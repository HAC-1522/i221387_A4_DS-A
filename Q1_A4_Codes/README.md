# Assignment 4 — Question 1: Knowledge Distillation on CIFAR-100

## Project overview

This project investigates knowledge distillation on CIFAR-100. A ResNet-50 teacher transfers knowledge to three smaller student architectures:

- MobileNetV2
- SqueezeNet 1.1
- EfficientNet-Lite0-style CIFAR-adapted MBConv model

Each student is trained independently as a baseline and again using knowledge distillation. The project supports hard-label cross-entropy, temperature-scaled soft-target distillation, and projected feature-based distillation. It also includes PyTorch inference, ONNX export, ONNX Runtime CPU inference, and benchmarking.

## 1. Project structure

| File or directory | Purpose |
|---|---|
| `models.py` | Teacher and student architectures |
| `train_teacher.py` | ResNet-50 teacher training |
| `train_baseline.py` | Independent student baseline training |
| `distill.py` | Student training with knowledge distillation |
| `train_utils.py` | Shared classifier training and evaluation |
| `common.py` | Shared dataset and utility functions |
| `inference.py` | PyTorch inference |
| `export_onnx.py` | ONNX model export |
| `onnx_inference.py` | ONNX Runtime CPU inference |
| `benchmark.py` | PyTorch CPU benchmark |
| `onnx_benchmark.py` | ONNX Runtime CPU benchmark |
| `artifacts/` | Trained model checkpoints |
| `results/` | Training histories and benchmark CSV files |
| `onnx/` | Exported ONNX models |

## 2. Environment and dependencies

The experiments were conducted using Python 3.13.16 and a Tesla T4 GPU.

Most training runs used PyTorch 2.14.1+cu130. The EfficientNet-Lite0 distilled run used PyTorch 2.11.0+cu130. The ONNX benchmark used ONNX Runtime with CPUExecutionProvider.

Recommended setup:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If ONNX export reports that `onnxscript` is missing, install it:

```bash
python -m pip install onnxscript
```

A CUDA-enabled PyTorch installation is recommended for training. CPU inference is used for deployment benchmarking.

## 3. Dataset

The project uses CIFAR-100:

- 50,000 training images
- 10,000 official test images
- 100 image classes
- RGB images with a resolution of 32 × 32

The training split is divided into 45,000 training images and 5,000 validation images. The official 10,000-image test split is kept separate for final evaluation.

The data loader downloads CIFAR-100 into `data/` when required. Standard CIFAR-100 channel normalization is applied.

## 4. Architectures

### Teacher

ResNet-50 is trained as the teacher model. Its pooled feature representation has 2,048 dimensions, followed by a 100-class classifier.

### Student models

- **MobileNetV2:** lightweight inverted residual architecture.
- **SqueezeNet 1.1:** compact architecture based on fire modules.
- **EfficientNet-Lite0-style:** a locally implemented CIFAR-adapted MBConv network using ReLU6 and depthwise convolutions, without squeeze-excitation.

The EfficientNet-Lite0 implementation is a project-specific adaptation, not an official pretrained EfficientNet-Lite0 implementation.

## 5. Training procedure

All teacher, baseline, and distilled student runs completed 30 epochs with batch size 128 and random seed 42.

| Setting | Teacher | Baseline students | Distilled students |
|---|---:|---:|---:|
| Epochs | 30 | 30 | 30 |
| Batch size | 128 | 128 | 128 |
| Learning rate | 0.1 | 0.05 | 0.05 |
| Optimizer | SGD | SGD | SGD |
| Momentum | 0.9 | 0.9 | 0.9 |
| Weight decay | 0.0005 | 0.0005 | 0.0005 |
| Nesterov | Yes | Yes | Yes |
| Scheduler | Cosine annealing | Cosine annealing | Cosine annealing |
| Random seed | 42 | 42 | 42 |

Baseline students are trained using cross-entropy loss only.

## 6. Knowledge distillation

The distilled students combine three objectives:

\[
L = (1-\alpha)L_{CE} + \alpha T^2L_{KL} + \lambda L_{feature}
\]

Where:

- \(L_{CE}\): hard-label cross-entropy loss.
- \(L_{KL}\): temperature-scaled KL divergence between teacher and student predictions.
- \(T\): temperature.
- \(\alpha\): soft-target loss weight.
- \(L_{feature}\): feature matching loss.
- \(\lambda\): feature loss weight.

Experiment settings:

| Parameter | Value |
|---|---:|
| Temperature | 4.0 |
| Alpha | 0.5 |
| Feature weight | 0.1 |
| Teacher feature dimension | 2,048 |
| MobileNetV2 feature dimension | 1,280 |

The student feature vector is projected to the teacher feature dimension. Both feature vectors are L2-normalized before feature matching. The projection is used during training; the deployed student checkpoint contains the student model rather than the training-only projection.

## 7. Training and evaluation results

All seven training runs completed 30 epochs.

### Best validation accuracy

| Model | Best validation accuracy | Best epoch |
|---|---:|---:|
| ResNet-50 teacher | 71.02% | 30 |
| MobileNetV2 baseline | 50.68% | 30 |
| SqueezeNet baseline | 13.02% | 29 |
| EfficientNet-Lite0 baseline | 52.32% | 29 |
| MobileNetV2 distilled | 54.60% | 29 |
| SqueezeNet distilled | 23.10% | 30 |
| EfficientNet-Lite0 distilled | 54.64% | 29 |

### Official CIFAR-100 test accuracy

| Student | Baseline | Distilled | Improvement |
|---|---:|---:|---:|
| MobileNetV2 | 50.63% | 54.31% | +3.68 percentage points |
| SqueezeNet | 13.12% | 22.10% | +8.98 percentage points |
| EfficientNet-Lite0-style | 51.48% | 54.46% | +2.98 percentage points |

All six student variants were evaluated on the 10,000 official CIFAR-100 test images. Distillation improved test accuracy for all three student architectures.

### PyTorch CPU benchmark

| Model | Test accuracy | Mean latency (ms/image) | Checkpoint size (MB) |
|---|---:|---:|---:|
| MobileNetV2 baseline | 50.63% | 0.497818 | 9.2095 |
| MobileNetV2 distilled | 54.31% | 0.490011 | 9.2099 |
| SqueezeNet baseline | 13.12% | 0.267358 | 2.9719 |
| SqueezeNet distilled | 22.10% | 0.265102 | 2.9720 |
| EfficientNet-Lite0 baseline | 51.48% | 2.188864 | 13.2191 |
| EfficientNet-Lite0 distilled | 54.46% | 2.202706 | 13.2194 |

Accuracy is calculated over all 10,000 test images. Warm-up batches are excluded from latency timing.

### ONNX Runtime CPU benchmark

| Model | Test accuracy | Mean latency (ms/image) | ONNX size (MB) |
|---|---:|---:|---:|
| MobileNetV2 baseline | 50.63% | 0.246143 | 9.1469 |
| MobileNetV2 distilled | 54.31% | 0.285765 | 9.1470 |
| SqueezeNet baseline | 13.12% | 0.077961 | 3.0611 |
| SqueezeNet distilled | 22.10% | 0.062564 | 3.0612 |
| EfficientNet-Lite0 baseline | 51.48% | 0.693442 | 13.0491 |
| EfficientNet-Lite0 distilled | 54.46% | 0.699685 | 13.0497 |

All six ONNX models were evaluated on the 10,000-image test set. The first 10 batches were excluded from latency timing, leaving 8,720 timed images. Accuracy is still calculated over all 10,000 images.

Latency measurements are specific to the benchmark runtime and CPU environment; they should not be treated as universal hardware performance figures.

## 8. Running the project

Run commands from the project root.

### A. Train the teacher

```bash
python train_teacher.py --epochs 30 --batch-size 128
```

### B. Train baseline students

```bash
python train_baseline.py --model mobilenetv2 --epochs 30 --batch-size 128
python train_baseline.py --model squeezenet --epochs 30 --batch-size 128
python train_baseline.py --model efficientnetlite0 --epochs 30 --batch-size 128
```

### C. Train distilled students

The teacher checkpoint must exist before running distillation.

```bash
python distill.py --model mobilenetv2 --epochs 30 --batch-size 128 --temperature 4 --alpha 0.5 --feature-weight 0.1
python distill.py --model squeezenet --epochs 30 --batch-size 128 --temperature 4 --alpha 0.5 --feature-weight 0.1
python distill.py --model efficientnetlite0 --epochs 30 --batch-size 128 --temperature 4 --alpha 0.5 --feature-weight 0.1
```

### D. PyTorch inference

```bash
python inference.py --model mobilenetv2_distilled --checkpoint artifacts/mobilenetv2_distilled.pt --image sample.png
```

Valid model names:

- `mobilenetv2_baseline`
- `squeezenet_baseline`
- `efficientnetlite0_baseline`
- `mobilenetv2_distilled`
- `squeezenet_distilled`
- `efficientnetlite0_distilled`

### E. ONNX export

Example:

```bash
python export_onnx.py --model mobilenetv2_baseline --checkpoint artifacts/mobilenetv2_baseline.pt --output onnx/mobilenetv2_baseline.onnx
```

Repeat for the other five model/checkpoint combinations.

### F. ONNX Runtime inference

```bash
python onnx_inference.py --model onnx/mobilenetv2_baseline.onnx --image sample.png
```

The inference script uses `CPUExecutionProvider`.

### G. Run benchmarks

PyTorch:

```bash
python benchmark.py --batch-size 128 --output results/student_benchmark.csv
```

ONNX Runtime:

```bash
python onnx_benchmark.py
```

The benchmark CSV files are saved under `results/`.

## 9. Limitations and implementation notes

- Training was performed under a limited compute budget using a Tesla T4 GPU.
- The teacher reached 71.02% best validation accuracy; student performance is consequently limited by the teacher and student architectures.
- SqueezeNet achieved substantially lower accuracy than the other students, although distillation improved its test accuracy.
- The EfficientNet-Lite0-style student is a locally implemented CIFAR-adapted architecture, not an official pretrained model.
- PyTorch versions differed between experiment sessions: most runs used 2.14.1+cu130, while EfficientNet-Lite0 distillation used 2.11.0+cu130.
- ONNX export and inference were completed for all six student variants. The EfficientNet-Lite0 distilled export retained opset 18 after an attempted conversion to opset 17 was unsuccessful; ONNX validation and inference still passed.
- CPU latency depends on the runtime and hardware and should be interpreted as a measured result of this environment.

## 10. Conclusion

Knowledge distillation improved the official CIFAR-100 test accuracy of all three student architectures. MobileNetV2 improved by 3.68 percentage points, SqueezeNet by 8.98 percentage points, and EfficientNet-Lite0-style by 2.98 percentage points.

The EfficientNet-Lite0-style distilled student achieved the highest test accuracy at 54.46%. SqueezeNet distilled achieved the lowest measured ONNX CPU latency at 0.062564 ms per image. All six student models were benchmarked in both PyTorch and ONNX Runtime, providing a comparison of accuracy, latency, and model size.
