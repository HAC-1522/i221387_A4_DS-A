# Assignment 4 — Question 2: Neural Architecture Search
## Final Technical Report

**Student:** ____________________  
**Roll Number:** ____________________  
**Course / Section:** ____________________  
**Instructor:** ____________________

## 1. Objective
Describe the development of an automated hardware-aware NAS framework that discovers lightweight CNNs for edge deployment.

## 2. Environment
| Item | Actual value |
|---|---|
| Platform | Kaggle |
| Python | Fill from execution |
| PyTorch | Fill from execution |
| GPU | Fill from execution |
| CPU | Fill from execution |
| OS/runtime | Fill from execution |

## 3. Dataset and preprocessing
Document CIFAR-10, training/validation split, normalization, augmentations, and variable input resolutions.

## 4. Search space and architecture generator
Explain variable depth, channels, kernels, standard/depthwise-separable blocks, activation functions, and resolution. Include the selected architecture JSON.

## 5. Evolutionary search strategy
Describe population initialization, candidate evaluation, fitness, elitism, mutation, crossover, generations, and reproducibility seed.

## 6. Training and validation pipeline
Report candidate training budget, final training schedule, optimizer, loss, batch size, and validation procedure.

## 7. Hardware-aware evaluation
| Metric | Measurement method | Constraint |
|---|---|---|
| CPU latency | Batch-1 CPU timing; warmup excluded | Fill actual limit |
| Model size | Serialized state-dictionary size | Fill actual limit |
| Memory | Process RSS snapshot when psutil is available | Fill actual limit |

Clarify that process RSS is not isolated model peak memory.

## 8. Search results
Insert the actual `results/search_history.csv` summary. Report total candidates, feasible candidates, rejected candidates and reasons.

## 9. Selected architecture and final evaluation
Report the selected configuration, best validation accuracy, official CIFAR-10 test accuracy, CPU latency, storage size, and memory measurement.

## 10. ONNX export and ONNX Runtime
Document ONNX checker result, input/output tensor shapes, ONNX Runtime CPU provider, test accuracy, latency/throughput, and any numerical differences checked.

## 11. Discussion and limitations
Discuss search-budget trade-offs, measurement variability, CPU dependence, process-level memory measurement, and whether the selected model satisfies all deployment constraints.

## 12. Conclusion
Summarize what was implemented and the measured outcome. Use only actual experiment results.
