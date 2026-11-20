"""Require all seeds/methods, verify predictions and report paired seed differences."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from waterbirds_core import (METHODS, best_epoch, detection_metrics, early_scores, group_metrics,
                             read_csv, select_examples, validate_config, write_csv,
                             write_json)


def summarise(root, config, allow_synthetic=False):
    root = Path(root)
    c = validate_config(config)
    rows, detections = [], []
    dataset_identity = None
    environments = None
    for seed in c["seeds"]:
        seed_root = root / f"seed_{seed}"
        if not (seed_root / "COMPLETE").is_file():
            raise ValueError(f"Incomplete seed {seed}")
        metadata = json.loads((seed_root / "metadata.json").read_text())
        identity = metadata["identity"]
        if identity["config"] != c or identity["seed"] != seed:
            raise ValueError("Config/seed mismatch")
        if identity["dataset"]["kind"] == "synthetic" and not allow_synthetic:
            raise ValueError("Synthetic smoke is not a scientific comparison")
        if dataset_identity is None:
            dataset_identity = identity["dataset"]
            environments = {k: identity[k] for k in ("versions", "source_hashes", "device", "weights_sha256", "python")}
        if identity["dataset"] != dataset_identity or any(identity[k] != v for k,v in environments.items()):
            raise ValueError("Dataset or experiment environment changed between seeds")
        dynamics = read_csv(seed_root / "erm" / "training_dynamics.csv")
        train_ids = sorted({int(r["sample_id"]) for r in dynamics})
        if len(train_ids) != dataset_identity["n_train"]:
            raise ValueError("Wrong number of training IDs")
        if any(int(r["seed"]) != seed for r in dynamics):
            raise ValueError("Dynamics seed mismatch")
        early_scores(dynamics, train_ids, c["epochs"])
        scores = early_scores(dynamics, train_ids, c["early_epoch"])
        initial_hash = None
        predictions_by_method = {}
        for method in METHODS:
            folder = seed_root / method
            if not (folder / "COMPLETE").is_file():
                raise ValueError(f"Incomplete method {method}")
            result = json.loads((folder / "result.json").read_text())
            signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if result["signature"] != signature:
                raise ValueError("Result provenance mismatch")
            if result["config"] != c or result["seed"] != seed or result["method"] != method:
                raise ValueError("Method config mismatch")
            if initial_hash is None:
                initial_hash = result["initial_model_sha256"]
            if result["initial_model_sha256"] != initial_hash:
                raise ValueError("Methods did not start from the same model")
            history = read_csv(folder / "epoch_metrics.csv")
            if len(history) != c["epochs"] or int(best_epoch(history)["epoch"]) != result["selected_epoch"]:
                raise ValueError("Incomplete history or incorrect checkpoint selection")
            selection = json.loads((folder / "selection.json").read_text())
            expected = select_examples(scores, method, c["fraction"], seed)
            if selection["selected_ids"] != expected or selection["subgroup_metadata_used"]:
                raise ValueError("Selection does not match prespecified training-only rule")
            row = dict(seed=seed, method=method, selected_epoch=result["selected_epoch"], selected_count=len(expected))
            for split in ("validation", "test"):
                predictions = read_csv(folder / f"{split}_predictions.csv")
                if len(predictions) != dataset_identity["n_val" if split == "validation" else "n_test"]:
                    raise ValueError("Missing predictions")
                metrics = group_metrics(predictions)
                if metrics != result[split]:
                    raise ValueError("Reported metrics disagree with predictions")
                if split == "validation" and metrics["overall_accuracy"] != float(best_epoch(history)["validation_accuracy"]):
                    raise ValueError("Selected validation accuracy does not match checkpoint history")
                row.update({split + "_" + k: v for k,v in metrics.items()})
                signature = sorted((int(r["sample_id"]), int(r["label"]), int(r["background"])) for r in predictions)
                key = (split, method)
                predictions_by_method[key] = signature
                if method != "erm" and signature != predictions_by_method[split, "erm"]:
                    raise ValueError("Evaluation examples changed between methods")
            test_ids = {i for i,_,_ in predictions_by_method["test",method]}
            val_ids = {i for i,_,_ in predictions_by_method["validation",method]}
            if test_ids & val_ids or (test_ids | val_ids) & set(train_ids):
                raise ValueError("Train/validation/test overlap")
            rows.append(row)
        train_metadata = read_csv(seed_root / "training_metadata.csv")
        detections.append(dict(seed=seed, **detection_metrics(scores, train_metadata)))
    summaries, paired = [], []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        row = dict(method=method, seeds=len(selected))
        for metric in ("test_overall_accuracy", "test_worst_group_accuracy"):
            values = [r[metric] for r in selected]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_sample_std"] = statistics.stdev(values) if len(values) > 1 else ""
        summaries.append(row)
        for r in selected:
            base = next(v for v in rows if v["method"] == "erm" and v["seed"] == r["seed"])
            paired.append(dict(seed=r["seed"], method=method,
                overall_delta_pp=100*(r["test_overall_accuracy"]-base["test_overall_accuracy"]),
                worst_group_delta_pp=100*(r["test_worst_group_accuracy"]-base["test_worst_group_accuracy"])))
    return rows, summaries, paired, detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--config", default="configs/comparison.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    rows, summary, paired, detections = summarise(args.run, config, args.allow_synthetic)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("per_seed",rows), ("summary",summary),
                         ("paired_differences",paired), ("early_detection",detections)):
        write_csv(output / f"{name}.csv", values)
    print(f"Validated {len(rows)} method/seed results")
    for r in summary:
        print(f"{r['method']:10} overall={r['test_overall_accuracy_mean']:.4f} "
              f"worst-group={r['test_worst_group_accuracy_mean']:.4f}")
    if not args.allow_synthetic:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
        for ax, metric, title in zip(axes, ("test_overall_accuracy", "test_worst_group_accuracy"),
                                     ("Overall test accuracy", "Worst-group test accuracy")):
            for seed in config["seeds"]:
                values = [next(r[metric] for r in rows if r["method"] == m and r["seed"] == seed) for m in METHODS]
                ax.plot(METHODS, values, marker="o", alpha=.65, label=f"seed {seed}")
            ax.set(title=title, ylabel="Accuracy", ylim=(0,1))
            ax.grid(alpha=.2)
        axes[-1].legend()
        fig.savefig(output / "comparison.png",dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
