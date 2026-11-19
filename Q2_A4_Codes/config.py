from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class SearchConfig:
    data_dir: str = "./data"
    output_dir: str = "."
    seed: int = 42
    population_size: int = 8
    generations: int = 5
    candidate_epochs: int = 3
    final_epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.05
    weight_decay: float = 5e-4
    num_workers: int = 2
    latency_limit_ms: float = 10.0
    model_size_limit_mb: float = 15.0
    memory_limit_mb: float = 2048.0
    latency_warmup: int = 10
    latency_iterations: int = 100
    train_subset: int = 12000
    val_subset: int = 3000
    device: str = "auto"

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))
