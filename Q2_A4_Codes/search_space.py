import random
from copy import deepcopy

DEPTHS = [2, 3, 4, 5, 6]
CHANNELS = [16, 24, 32, 48, 64, 96, 128]
KERNELS = [3, 5]
CONV_TYPES = ["standard", "depthwise"]
ACTIVATIONS = ["relu", "relu6", "gelu", "silu"]
RESOLUTIONS = [24, 28, 32, 40]

def random_architecture():
    depth = random.choice(DEPTHS)
    return {
        "depth": depth,
        "channels": [random.choice(CHANNELS) for _ in range(depth)],
        "kernels": [random.choice(KERNELS) for _ in range(depth)],
        "conv_types": [random.choice(CONV_TYPES) for _ in range(depth)],
        "activations": [random.choice(ACTIVATIONS) for _ in range(depth)],
        "input_resolution": random.choice(RESOLUTIONS),
    }

def repair_architecture(cfg):
    """Normalize a possibly mutated config into a valid architecture."""
    depth = max(min(int(cfg.get("depth", 3)), max(DEPTHS)), min(DEPTHS))
    out = {"depth": depth}
    choices = {
        "channels": CHANNELS,
        "kernels": KERNELS,
        "conv_types": CONV_TYPES,
        "activations": ACTIVATIONS,
    }
    for key, allowed in choices.items():
        values = list(cfg.get(key, []))
        values = (values + [random.choice(allowed)] * depth)[:depth]
        out[key] = [v if v in allowed else random.choice(allowed) for v in values]
    resolution = cfg.get("input_resolution", 32)
    out["input_resolution"] = resolution if resolution in RESOLUTIONS else 32
    return out

def mutate(cfg):
    child = deepcopy(cfg)
    operation = random.choice(["depth", "channel", "kernel", "type", "activation", "resolution"])
    if operation == "depth":
        child["depth"] = random.choice(DEPTHS)
    elif operation == "resolution":
        child["input_resolution"] = random.choice(RESOLUTIONS)
    else:
        depth = child["depth"]
        i = random.randrange(depth)
        if operation == "channel":
            child["channels"][i] = random.choice(CHANNELS)
        elif operation == "kernel":
            child["kernels"][i] = random.choice(KERNELS)
        elif operation == "type":
            child["conv_types"][i] = random.choice(CONV_TYPES)
        elif operation == "activation":
            child["activations"][i] = random.choice(ACTIVATIONS)
    return repair_architecture(child)

def crossover(a, b):
    depth = random.choice([a["depth"], b["depth"]])
    child = {"depth": depth}
    for key in ["channels", "kernels", "conv_types", "activations"]:
        child[key] = [
            random.choice([a[key][i % a["depth"]], b[key][i % b["depth"]]])
            for i in range(depth)
        ]
    child["input_resolution"] = random.choice([a["input_resolution"], b["input_resolution"]])
    return repair_architecture(child)
