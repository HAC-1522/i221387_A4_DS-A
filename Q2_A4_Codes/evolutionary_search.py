import json, random, hashlib
from pathlib import Path
import pandas as pd
import torch
from models import SearchableCNN
from search_space import random_architecture, mutate, crossover
from data import build_loaders
from engine import fit, get_device
from hardware import measure_hardware, check_constraints

def arch_id(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]

def evaluate_candidate(cfg, args, device):
    train_loader, val_loader, _ = build_loaders(
        args.data_dir, cfg["input_resolution"], args.batch_size,
        args.num_workers, args.train_subset, args.val_subset, args.seed
    )
    model = SearchableCNN(cfg)
    model, history, best_val = fit(
        model, train_loader, val_loader, args.candidate_epochs, device,
        args.learning_rate, args.weight_decay
    )
    metrics = measure_hardware(
        model, (1, 3, cfg["input_resolution"], cfg["input_resolution"]),
        args.latency_warmup, args.latency_iterations
    )
    feasible, reason = check_constraints(
        metrics, args.latency_limit_ms, args.model_size_limit_mb, args.memory_limit_mb
    )
    # Accuracy is the primary objective among feasible candidates.
    fitness = best_val if feasible else -1.0
    return {
        "architecture": cfg, "architecture_id": arch_id(cfg),
        "validation_accuracy": best_val, "fitness": fitness,
        "feasible": feasible, "rejection_reason": reason, **metrics
    }

def run_search(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    results_dir = Path(args.output_dir) / "results"
    artifacts_dir = Path(args.output_dir) / "artifacts"
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    population = [random_architecture() for _ in range(args.population_size)]
    evaluated = {}
    rows = []
    best = None

    # Resume previously completed candidates when history exists.
    history_file = results_dir / "search_history.csv"

    if history_file.exists():
        previous = pd.read_csv(history_file)

        for _, saved in previous.iterrows():
            record = saved.to_dict()
            record["architecture"] = json.loads(record["architecture"])
            record["architecture_id"] = str(record["architecture_id"])
            record["generation"] = int(record["generation"])
            record["feasible"] = str(record["feasible"]).lower() == "true"

            evaluated[record["architecture_id"]] = record
            rows.append(record)

            if record["feasible"]:
                if best is None or record["fitness"] > best["fitness"]:
                    best = record

        print(
            f"RESUME: Loaded {len(evaluated)} previously evaluated candidates."
        )

    for generation in range(args.generations):
        print(f"\n========== GENERATION {generation+1}/{args.generations} ==========")
        for cfg in population:
            key = arch_id(cfg)
            if key in evaluated:
                continue
            print(f"\nEvaluating {key}: {cfg}")
            record = evaluate_candidate(cfg, args, device)
            record["generation"] = generation + 1
            evaluated[key] = record
            rows.append(record)
            if record["feasible"] and (best is None or record["fitness"] > best["fitness"]):
                best = record
                with open(artifacts_dir / "best_architecture.json", "w") as f:
                    json.dump(best["architecture"], f, indent=2)
            pd.DataFrame([{k: json.dumps(v) if k == "architecture" else v
                           for k, v in r.items()} for r in rows]).to_csv(
                               results_dir / "search_history.csv", index=False)

        feasible = [r for r in evaluated.values() if r["feasible"]]
        if not feasible:
            # Keep searching from the least-bad accuracy candidates; final report will
            # clearly indicate if no architecture met the deployment limits.
            ranked = sorted(evaluated.values(), key=lambda r: r["validation_accuracy"], reverse=True)
        else:
            ranked = sorted(feasible, key=lambda r: r["fitness"], reverse=True)
        elites = [r["architecture"] for r in ranked[:max(2, min(4, len(ranked)))]]
        next_population = elites[:]
        while len(next_population) < args.population_size:
            if len(elites) >= 2 and random.random() < 0.5:
                p1, p2 = random.sample(elites, 2)
                child = crossover(p1, p2)
            else:
                child = mutate(random.choice(elites))
            next_population.append(child)
        population = next_population

    if best is None:
        # Save the best observed candidate for inspection, but do not silently call it feasible.
        best = max(evaluated.values(), key=lambda r: r["validation_accuracy"])
        best["feasible"] = False
        best["selection_warning"] = "No candidate satisfied all configured deployment constraints."
    with open(artifacts_dir / "best_search_record.json", "w") as f:
        json.dump(best, f, indent=2)
    pd.DataFrame([{k: json.dumps(v) if k == "architecture" else v
                   for k, v in r.items()} for r in rows]).to_csv(
                       results_dir / "search_history.csv", index=False)
    print("\nSearch complete.")
    print("Best architecture:", best["architecture"])
    print("Validation accuracy:", best["validation_accuracy"])
    print("Feasible:", best["feasible"])
    return best
