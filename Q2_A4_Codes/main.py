import argparse
from config import SearchConfig
from engine import seed_everything
from evolutionary_search import run_search

def main():
    p = argparse.ArgumentParser(description="Hardware-aware evolutionary NAS for CIFAR-10")
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--population-size", type=int, default=None)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--candidate-epochs", type=int, default=None)
    p.add_argument("--latency-limit-ms", type=float, default=None)
    p.add_argument("--model-size-limit-mb", type=float, default=None)
    p.add_argument("--memory-limit-mb", type=float, default=None)
    args = p.parse_args()

    cfg = SearchConfig.load(args.config) if args.config else SearchConfig()
    for key in ["data_dir", "output_dir", "population_size", "generations",
                "candidate_epochs", "latency_limit_ms", "model_size_limit_mb",
                "memory_limit_mb"]:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    seed_everything(cfg.seed)
    cfg.save(f"{cfg.output_dir}/search_config.json")
    run_search(cfg)

if __name__ == "__main__":
    main()
