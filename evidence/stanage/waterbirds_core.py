"""Data-only experiment rules. No PyTorch, dataset download or training on import."""
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

METHODS = ("erm", "jtt", "early_loss", "random")


def validate_config(c):
    if c.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    for name in ("epochs", "early_epoch", "batch_size", "image_size"):
        if type(c.get(name)) is not int or c[name] < 1:
            raise ValueError(f"Invalid {name}")
    if c["early_epoch"] > c["epochs"]:
        raise ValueError("early_epoch exceeds training budget")
    if type(c.get("num_workers")) is not int or c["num_workers"] < 0:
        raise ValueError("Invalid num_workers")
    for name in ("learning_rate", "weight_decay", "upweight", "fraction"):
        if not isinstance(c.get(name), (int, float)) or not math.isfinite(c[name]):
            raise ValueError(f"Invalid {name}")
    if (c["learning_rate"] <= 0 or c["weight_decay"] < 0 or c["upweight"] < 1
            or not 0 < c["fraction"] <= 1):
        raise ValueError("Invalid weighting or optimiser settings")
    seeds = c.get("seeds", [])
    if not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique unsigned 32-bit integers")
    if c.get("methods") != list(METHODS):
        raise ValueError("Run the matched erm, jtt, early_loss, random comparison")
    if type(c.get("pretrained")) is not bool:
        raise ValueError("pretrained must be explicit")
    return c


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export empty table")
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def early_scores(rows, train_ids, epoch):
    """Whitelist ID/loss/correctness; never inspect group/background metadata."""
    train_ids = list(map(int, train_ids))
    if not train_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError("Training IDs must be nonempty and unique")
    wanted = set(train_ids)
    values = defaultdict(dict)
    seeds = set()
    for row in rows:
        e = int(row["epoch"])
        if e > epoch:
            continue
        sample = int(row["sample_id"])
        loss, correct = float(row["per_example_loss"]), int(row["correct"])
        if sample not in wanted or not 1 <= e <= epoch or not math.isfinite(loss) or loss < 0 or correct not in (0, 1):
            raise ValueError("Invalid training dynamics")
        if e in values[sample]:
            raise ValueError("Duplicate sample/epoch")
        seeds.add(int(row["seed"]))
        values[sample][e] = (loss, correct)
    if len(seeds) != 1 or set(values) != wanted:
        raise ValueError("Mixed seeds or missing training IDs")
    result = []
    for sample in sorted(wanted):
        v = values[sample]
        if set(v) != set(range(1, epoch + 1)):
            raise ValueError("Incomplete early training window")
        result.append(dict(sample_id=sample, mean_loss=sum(x[0] for x in v.values()) / epoch,
                           early_error=1-v[epoch][1]))
    return result


def select_examples(scores, method, fraction, seed):
    ids = sorted(int(r["sample_id"]) for r in scores)
    if not ids or len(ids) != len(set(ids)) or not 0 < fraction <= 1:
        raise ValueError("Invalid selection inputs")
    k = max(1, math.ceil(len(ids) * fraction))
    if method == "erm":
        return []
    if method == "jtt":
        return sorted(int(r["sample_id"]) for r in scores if r["early_error"])
    if method == "early_loss":
        return [int(r["sample_id"]) for r in sorted(scores, key=lambda r: (-r["mean_loss"], r["sample_id"]))[:k]]
    if method == "random":
        return sorted(random.Random(seed).sample(ids, k))
    raise ValueError("Unknown method")


def sample_weights(train_ids, selected_ids, upweight):
    ids, selected = list(map(int, train_ids)), set(map(int, selected_ids))
    if len(ids) != len(set(ids)) or not selected.issubset(ids) or upweight < 1 or not math.isfinite(upweight):
        raise ValueError("Invalid sample weights")
    return [float(upweight) if i in selected else 1.0 for i in ids]


def group_metrics(predictions):
    """Raw overall accuracy, plus all four bird/background groups at evaluation."""
    if not predictions:
        raise ValueError("Empty predictions")
    groups = {g: [] for g in range(4)}
    seen = set()
    for row in predictions:
        sample = int(row["sample_id"])
        y, pred, bg = (int(row[k]) for k in ("label", "prediction", "background"))
        if sample in seen or y not in (0, 1) or pred not in (0, 1) or bg not in (0, 1):
            raise ValueError("Invalid predictions")
        seen.add(sample)
        groups[2*bg + y].append(int(pred == y))
    if any(not v for v in groups.values()):
        raise ValueError("Cannot report worst-group accuracy with a missing group")
    result = dict(n=len(predictions), overall_accuracy=sum(sum(v) for v in groups.values()) / len(predictions))
    for g, values in groups.items():
        result[f"group_{g}_n"] = len(values)
        result[f"group_{g}_accuracy"] = sum(values) / len(values)
    result["worst_group_accuracy"] = min(result[f"group_{g}_accuracy"] for g in groups)
    return result


def best_epoch(history):
    if not history:
        raise ValueError("Missing epoch history")
    for i, row in enumerate(history, 1):
        if int(row["epoch"]) != i or not 0 <= float(row["validation_accuracy"]) <= 1:
            raise ValueError("Invalid epoch history")
    # Strict improvement and earliest tie. No subgroup or test metrics accessed.
    return max(history, key=lambda r: (float(r["validation_accuracy"]), -int(r["epoch"])))


def detection_metrics(scores, metadata):
    """Retrospective early-loss detection with tie-aware ROC-AUC and AP."""
    lookup = {int(r["sample_id"]): int(r["label"]) != int(r["background"]) for r in metadata}
    if len(lookup) != len(metadata) or set(lookup) != {r["sample_id"] for r in scores}:
        raise ValueError("Detection metadata IDs do not match scores")
    ordered = sorted(scores, key=lambda r: (r["mean_loss"], r["sample_id"]))
    positives = sum(lookup.values())
    negatives = len(lookup)-positives
    if not positives or not negatives:
        raise ValueError("Detection requires both conflicting and aligned examples")
    ties = defaultdict(list)
    for r in ordered:
        ties[r["mean_loss"]].append(lookup[r["sample_id"]])
    lower_negatives, concordance = 0, 0.0
    for loss in sorted(ties):
        group = ties[loss]
        p, n = sum(group), len(group)-sum(group)
        concordance += p*(lower_negatives + .5*n)
        lower_negatives += n
    tp, count, ap = 0, 0, 0.0
    for loss in sorted(ties, reverse=True):
        group = ties[loss]
        tp += sum(group)
        count += len(group)
        ap += sum(group)/positives * tp/count
    k = max(1, math.ceil(.1*len(ordered)))
    top = sorted(scores, key=lambda r: (-r["mean_loss"], r["sample_id"]))[:k]
    return dict(n=len(lookup), conflicting_count=positives, prevalence=positives/len(lookup),
                roc_auc=concordance/(positives*negatives), average_precision=ap,
                precision_at_10pct=sum(lookup[r["sample_id"]] for r in top)/k)
