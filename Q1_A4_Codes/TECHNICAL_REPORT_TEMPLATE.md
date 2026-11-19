# Assignment 4 — Question 1 Technical Report

## 1. Objective

The objective of this experiment is to study knowledge distillation on CIFAR-100 by transferring information from a ResNet-50 teacher to three smaller student architectures: MobileNetV2, SqueezeNet 1.1, and an EfficientNet-Lite0-style CIFAR-adapted model.

Each student is trained independently as a baseline and then trained using a combination of hard-label supervision, temperature-scaled soft-target distillation, and feature-based distillation. The resulting models are evaluated using the official CIFAR-100 test set and compared using PyTorch and ONNX Runtime CPU benchmarks.

## 2. Environment

| Item | Actual value |
|---|---|
| Python version | 3.13.16 |
| Training GPU | NVIDIA Tesla T4 |
| Training framework | PyTorch |
| PyTorch version | 2.14.1+cu130 for teacher, baseline students, and MobileNetV2/SqueezeNet distillation |
| PyTorch version exception | 2.11.0+cu130 for EfficientNet-Lite0 distillation |
| Evaluation device | CPU |
| ONNX execution provider | CPUExecutionProvider |
| Random seed | 42 |
| Epochs per training run | 30 |
| Batch size | 128 |

The experiments were completed across multiple runtime sessions. Therefore, the PyTorch version difference is documented rather than presenting all runs as if they used one identical software environment.

## 3. Dataset and preprocessing

CIFAR-100 was used for training and evaluation.

- Total training images: 50,000
- Training subset: 45,000 images
- Validation subset: 5,000 images
- Official test images: 10,000
- Number of classes: 100
- Image dimensions: 32 × 32 RGB

The 50,000-image training split was divided into training and validation subsets. The official 10,000-image test set was kept separate for final evaluation. Standard CIFAR-100 channel mean and standard deviation normalization was applied.

## 4. Architectures

### 4.1 Teacher: ResNet-50

ResNet-50 was trained as the teacher model. Its classifier produces 100 class logits. Its pooled feature representation has 2,048 dimensions.

The teacher completed 30 epochs and achieved:
- Final training accuracy: 93.27%
- Best validation accuracy: 71.02%
- Best validation epoch: 30

### 4.2 Student: MobileNetV2

MobileNetV2 was used as a lightweight student architecture. Its pooled feature representation has 1,280 dimensions.

### 4.3 Student: SqueezeNet 1.1

SqueezeNet 1.1 was used as a compact student architecture based on fire modules.

### 4.4 Student: EfficientNet-Lite0-style

The project implements a local CIFAR-adapted Lite0-style MBConv network using ReLU6 activations and depthwise convolutions, without squeeze-excitation. It is a project-specific adaptation and should not be described as an official pretrained EfficientNet-Lite0 model.

## 5. Training methodology

All seven training runs completed 30 epochs with a batch size of 128 and random seed 42.

| Training setting | Teacher | Baseline students | Distilled students |
|---|---:|---:|---:|
| Epochs | 30 | 30 | 30 |
| Batch size | 128 | 128 | 128 |
| Initial learning rate | 0.1 | 0.05 | 0.05 |
| Optimizer | SGD | SGD | SGD |
| Momentum | 0.9 | 0.9 | 0.9 |
| Weight decay | 0.0005 | 0.0005 | 0.0005 |
| Nesterov | Enabled | Enabled | Enabled |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| Loss | Cross-entropy | Cross-entropy | Combined KD objective |

Baseline students are trained using cross-entropy loss. Distilled students use the combined objective described in Section 6.

## 6. Knowledge distillation

The student objective combines hard-label cross-entropy, soft-target KL divergence, and feature matching:

\[
L = (1-\alpha)L_{CE} + \alpha T^2L_{KL} + \lambda L_{feature}
\]

The terms represent:

- \(L_{CE}\): cross-entropy between student predictions and ground-truth labels.
- \(L_{KL}\): KL divergence between temperature-scaled teacher and student predictions.
- \(T\): distillation temperature.
- \(\alpha\): weight assigned to soft-target distillation.
- \(L_{feature}\): feature matching loss.
- \(\lambda\): feature matching weight.

The experiment used:

| Parameter | Value |
|---|---:|
| Temperature \(T\) | 4.0 |
| Alpha \(\alpha\) | 0.5 |
| Feature weight \(\lambda\) | 0.1 |
| Teacher feature dimension | 2,048 |
| MobileNetV2 feature dimension | 1,280 |

The student feature vector is projected to the teacher feature dimension. Both feature vectors are L2-normalized before matching. The projection is part of the training procedure and is not required for student inference.

## 7. Results

### 7.1 Training validation results

The following values are the best validation accuracies recorded in the training histories.

| Model | Best validation accuracy | Best epoch |
|---|---:|---:|
| ResNet-50 teacher | 71.02% | 30 |
| MobileNetV2 baseline | 50.68% | 30 |
| SqueezeNet baseline | 13.02% | 29 |
| EfficientNet-Lite0-style baseline | 52.32% | 29 |
| MobileNetV2 distilled | 54.60% | 29 |
| SqueezeNet distilled | 23.10% | 30 |
| EfficientNet-Lite0-style distilled | 54.64% | 29 |

### 7.2 Official test accuracy comparison

All six student variants were evaluated on the complete official CIFAR-100 test set containing 10,000 images.

| Student | Baseline test accuracy | Distilled test accuracy | Improvement |
|---|---:|---:|---:|
| MobileNetV2 | 50.63% | 54.31% | +3.68 percentage points |
| SqueezeNet | 13.12% | 22.10% | +8.98 percentage points |
| EfficientNet-Lite0-style | 51.48% | 54.46% | +2.98 percentage points |

Distillation improved test accuracy for all three student architectures. The largest absolute improvement was observed for SqueezeNet, at 8.98 percentage points.

### 7.3 PyTorch CPU benchmark

| Model | Test accuracy | Mean CPU latency (ms/image) | Checkpoint size (MB) |
|---|---:|---:|---:|
| MobileNetV2 baseline | 50.63% | 0.497818 | 9.2095 |
| MobileNetV2 distilled | 54.31% | 0.490011 | 9.2099 |
| SqueezeNet baseline | 13.12% | 0.267358 | 2.9719 |
| SqueezeNet distilled | 22.10% | 0.265102 | 2.9720 |
| EfficientNet-Lite0-style baseline | 51.48% | 2.188864 | 13.2191 |
| EfficientNet-Lite0-style distilled | 54.46% | 2.202706 | 13.2194 |

Accuracy was calculated over all 10,000 test images. Warm-up batches were excluded from latency timing. The latency values are measurements from the experiment's CPU runtime and are hardware-dependent.

### 7.4 ONNX Runtime CPU benchmark

| Model | Test accuracy | Mean CPU latency (ms/image) | ONNX model size (MB) |
|---|---:|---:|---:|
| MobileNetV2 baseline | 50.63% | 0.246143 | 9.1469 |
| MobileNetV2 distilled | 54.31% | 0.285765 | 9.1470 |
| SqueezeNet baseline | 13.12% | 0.077961 | 3.0611 |
| SqueezeNet distilled | 22.10% | 0.062564 | 3.0612 |
| EfficientNet-Lite0-style baseline | 51.48% | 0.693442 | 13.0491 |
| EfficientNet-Lite0-style distilled | 54.46% | 0.699685 | 13.0497 |

All six ONNX models were evaluated on the complete 10,000-image test set. The first 10 batches were excluded from latency timing, leaving 8,720 timed images. Accuracy still includes all 10,000 images.

The distilled SqueezeNet had the lowest measured ONNX CPU latency, at 0.062564 ms per image. The EfficientNet-Lite0-style distilled model achieved the highest test accuracy, at 54.46%.

## 8. ONNX deployment and validation

All six student variants were exported to ONNX and evaluated using ONNX Runtime with CPUExecutionProvider.

The ONNX benchmark completed for all six variants. The measured ONNX test accuracies matched the corresponding PyTorch test accuracies.

For the EfficientNet-Lite0 distilled model, export retained ONNX opset 18 after an attempted conversion to opset 17 was unsuccessful. The exported model passed ONNX checker validation and ONNX Runtime inference validation.

Additional parity validation for EfficientNet-Lite0 distilled produced:
- PyTorch predicted class: 39
- ONNX predicted class: 39
- Maximum absolute output difference: approximately 3.81 × 10⁻⁶
- Mean absolute output difference: approximately 6.82 × 10⁻⁷
- Predicted classes matched.

## 9. Limitations

1. **Compute budget:** Training was performed using a Tesla T4 GPU and a fixed 30-epoch budget. Longer training or additional hyperparameter tuning may change the results.
2. **Teacher performance:** The teacher achieved 71.02% best validation accuracy, which limits the quality of the information available to students.
3. **SqueezeNet performance:** SqueezeNet achieved lower accuracy than the other students, even after distillation. However, it also had the smallest ONNX model size and lowest measured ONNX CPU latency.
4. **Architecture adaptation:** The EfficientNet-Lite0-style model is a locally implemented CIFAR-adapted MBConv architecture rather than an official pretrained Lite0 model.
5. **Software environment:** Most runs used PyTorch 2.14.1+cu130, while EfficientNet-Lite0 distillation used PyTorch 2.11.0+cu130. This difference is documented for reproducibility.
6. **Export compatibility:** The EfficientNet-Lite0 distilled model retained opset 18 because the attempted opset 17 conversion did not succeed. ONNX validation and inference nevertheless passed.
7. **Hardware dependence:** CPU latency measurements depend on the CPU, runtime, and execution environment. They should not be generalized to other machines without benchmarking.

## 10. Conclusion

This experiment implemented a complete knowledge-distillation pipeline using a ResNet-50 teacher and three student architectures on CIFAR-100. All three distilled students improved over their independently trained baselines on the official test set.

MobileNetV2 improved by 3.68 percentage points, SqueezeNet by 8.98 percentage points, and EfficientNet-Lite0-style by 2.98 percentage points. The EfficientNet-Lite0-style distilled model achieved the highest test accuracy at 54.46%. SqueezeNet distilled achieved the lowest measured ONNX CPU latency at 0.062564 ms per image.

The project also completed PyTorch and ONNX Runtime CPU benchmarking for all six student variants. These results demonstrate the accuracy improvements obtained through distillation and provide measured comparisons of model size and CPU inference latency.
