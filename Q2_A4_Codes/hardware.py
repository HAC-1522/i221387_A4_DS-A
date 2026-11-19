import os, time, tempfile
from pathlib import Path
import torch

def measure_hardware(model, input_shape, warmup=10, iterations=100, device="cpu"):
    """Measure batch-1 CPU latency, serialized state size, and peak RSS if psutil exists."""
    import copy
    model = copy.deepcopy(model).cpu().eval()
    x = torch.randn(*input_shape, device="cpu")
    with torch.inference_mode():
        for _ in range(max(warmup, 1)):
            model(x)
        timings = []
        for _ in range(max(iterations, 1)):
            start = time.perf_counter()
            model(x)
            timings.append((time.perf_counter() - start) * 1000.0)
    size_mb = sum(t.numel() * t.element_size() for t in model.state_dict().values()) / (1024**2)
    # Also measure actual serialized checkpoint bytes, which is the deployment storage metric.
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name
    try:
        torch.save(model.state_dict(), temp_path)
        storage_mb = os.path.getsize(temp_path) / (1024**2)
    finally:
        os.remove(temp_path)

    peak_memory_mb = None
    try:
        import psutil
        process = psutil.Process(os.getpid())
        peak_memory_mb = process.memory_info().rss / (1024**2)
    except Exception:
        pass

    timings.sort()
    median_latency = timings[len(timings)//2]
    return {
        "cpu_latency_ms": float(median_latency),
        "mean_cpu_latency_ms": float(sum(timings) / len(timings)),
        "model_size_mb": float(storage_mb),
        "parameter_tensor_mb": float(size_mb),
        "peak_process_rss_mb": peak_memory_mb,
        "latency_measurement": "batch=1, CPU, median over timed iterations",
        "memory_measurement": "process RSS snapshot; not isolated model peak" if peak_memory_mb is not None else "unsupported"
    }

def check_constraints(metrics, latency_limit_ms, size_limit_mb, memory_limit_mb):
    reasons = []
    if metrics["cpu_latency_ms"] > latency_limit_ms:
        reasons.append("CPU latency limit exceeded")
    if metrics["model_size_mb"] > size_limit_mb:
        reasons.append("model storage limit exceeded")
    mem = metrics.get("peak_process_rss_mb")
    if memory_limit_mb is not None and mem is not None and mem > memory_limit_mb:
        reasons.append("process RSS limit exceeded")
    return len(reasons) == 0, "; ".join(reasons)
