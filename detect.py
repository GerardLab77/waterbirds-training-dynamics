"""Fixed exploratory detector comparison using existing ERM histories; no training."""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyse import summarise
from waterbirds_core import (detection_metrics, early_scores, file_sha, read_csv,
                             write_csv, write_json)

DETECTORS = ("mean_loss", "class_relative_loss")


def percentile_ranks(values):
    """Within-class midranks in [0,1]. Ties stay tied; singleton -> 0.5."""
    if not values or any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Scores must be finite and nonempty")
    groups = defaultdict(list)
    for sample, value in values.items():
        groups[value].append(sample)
    result, lower = {}, 0
    n = len(values)
    for value in sorted(groups):
        samples = groups[value]
        rank = (lower + (len(samples)-1)/2)/(n-1) if n > 1 else .5
        result.update({sample: rank for sample in samples})
        lower += len(samples)
    return result


def score_examples(dynamics, train_ids, epoch):
    """Only bird-type training labels, IDs and early losses enter scoring."""
    train_ids = list(map(int,train_ids))
    scores = early_scores(dynamics, train_ids, epoch)
    labels = {}
    for row in dynamics:
        if int(row["epoch"]) > epoch:
            continue
        sample, label = int(row["sample_id"]), int(row["label"])
        if label not in (0,1) or (sample in labels and labels[sample] != label):
            raise ValueError("Inconsistent training labels")
        labels[sample] = label
    if set(labels) != set(map(int,train_ids)) or set(labels.values()) != {0,1}:
        raise ValueError("Both bird classes and all training IDs are required")
    ranks = {}
    for label in (0,1):
        ranks.update(percentile_ranks({r["sample_id"]:r["mean_loss"] for r in scores
                                       if labels[r["sample_id"]] == label}))
    return [dict(sample_id=r["sample_id"], label=labels[r["sample_id"]],
                 mean_loss=r["mean_loss"], class_relative_loss=ranks[r["sample_id"]])
            for r in scores]


def evaluate(scores, metadata, detector):
    if detector not in DETECTORS:
        raise ValueError("Unknown detector")
    lookup = {int(r["sample_id"]):r for r in metadata}
    if len(lookup) != len(metadata) or set(lookup) != {r["sample_id"] for r in scores}:
        raise ValueError("Metadata IDs do not match detector IDs")
    for row in scores:
        meta = lookup[row["sample_id"]]
        if int(meta["label"]) != row["label"] or int(meta["background"]) not in (0,1):
            raise ValueError("Retrospective metadata disagree with training labels")
    # Historical metric helper calls its generic ranking input 'mean_loss'.
    metric_scores = [dict(sample_id=r["sample_id"],mean_loss=r[detector]) for r in scores]
    metrics = detection_metrics(metric_scores,metadata)
    k = max(1,math.ceil(.1*len(scores)))
    selected = sorted(scores,key=lambda r:(-r[detector],r["sample_id"]))[:k]
    found = sum(int(lookup[r["sample_id"]]["background"]) != r["label"] for r in selected)
    metrics.update(selected_count=k, selected_conflicting=found,
                   recall_at_10pct=found/metrics["conflicting_count"],
                   lift_at_10pct=metrics["precision_at_10pct"]/metrics["prevalence"],
                   selected_class_0=sum(r["label"] == 0 for r in selected),
                   selected_class_1=sum(r["label"] == 1 for r in selected))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--config",default="configs/comparison.json")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    root, output = Path(args.run), Path(args.output)
    config = json.loads(Path(args.config).read_text())
    # Preserve and validate the original comparison before any new reporting.
    summarise(root, config)
    metrics, all_scores, differences = [], [], []
    inputs = {}
    for seed in config["seeds"]:
        folder = root/f"seed_{seed}"
        dynamics = read_csv(folder/"erm/training_dynamics.csv")
        train_ids = sorted({int(r["sample_id"]) for r in dynamics})
        scores = score_examples(dynamics,train_ids,config["early_epoch"])
        # Backgrounds are first read after both detector scores are fixed.
        metadata = read_csv(folder/"training_metadata.csv")
        all_scores.extend(dict(seed=seed,**r) for r in scores)
        global_rows = {}
        for detector in DETECTORS:
            for label in (None,0,1):
                scoped = scores if label is None else [r for r in scores if r["label"] == label]
                selected_ids = {r["sample_id"] for r in scoped}
                meta = [r for r in metadata if int(r["sample_id"]) in selected_ids]
                row = dict(seed=seed,detector=detector,
                           scope="all" if label is None else f"bird_class_{label}",
                           **evaluate(scoped,meta,detector))
                metrics.append(row)
                if label is None:
                    global_rows[detector] = row
        base, candidate = (global_rows[d] for d in DETECTORS)
        differences.append(dict(seed=seed,
            precision_at_10pct_delta_pp=100*(candidate["precision_at_10pct"]-base["precision_at_10pct"]),
            average_precision_delta=candidate["average_precision"]-base["average_precision"],
            roc_auc_delta=candidate["roc_auc"]-base["roc_auc"]))
        inputs[str(seed)] = {name:file_sha(folder/name) for name in
                            ("erm/training_dynamics.csv","training_metadata.csv","metadata.json")}
    summary = []
    for detector in DETECTORS:
        values = [r for r in metrics if r["detector"] == detector and r["scope"] == "all"]
        row = dict(detector=detector,seeds=len(values))
        for metric in ("precision_at_10pct","average_precision","roc_auc","recall_at_10pct","lift_at_10pct"):
            data = [r[metric] for r in values]
            row[metric+"_mean"] = statistics.mean(data)
            row[metric+"_sample_std"] = statistics.stdev(data) if len(data)>1 else ""
        summary.append(row)
    if output.exists():
        raise ValueError("Use a fresh detector output directory to retain previous findings")
    output.mkdir(parents=True)
    write_csv(output/"scores.csv",all_scores)
    write_csv(output/"metrics.csv",metrics)
    write_csv(output/"summary.csv",summary)
    write_csv(output/"paired_differences.csv",differences)
    write_json(output/"protocol.json",dict(
        status="exploratory follow-up after inspecting the initial comparison",
        detectors=DETECTORS,epoch=config["early_epoch"],fraction=.1,
        primary_metric="precision_at_10pct",source_sha256=file_sha(__file__),
        input_hashes=inputs,subgroup_metadata_used_in_scoring=False,
        warning="Seeds reuse the same images; no held-out generalisation claim."))
    for row in summary:
        print(f"{row['detector']:22} top10_precision={row['precision_at_10pct_mean']:.4f} "
              f"AP={row['average_precision_mean']:.4f} AUC={row['roc_auc_mean']:.4f}")
    for row in differences:
        print(f"seed={row['seed']} class-relative minus raw: "
              f"top10 precision {row['precision_at_10pct_delta_pp']:+.2f} pp; "
              f"AP {row['average_precision_delta']:+.4f}")


if __name__ == "__main__":
    main()
