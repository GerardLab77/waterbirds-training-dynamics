"""Frozen selection rules and reference checks for class-relative retraining."""
import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from detect import score_examples
from waterbirds_core import (METHODS, best_epoch, early_scores, file_sha, read_csv,
                             validate_config, write_json)

RELATIVE_METHODS = ("class_relative", "class_matched_random")
TRAINING_FIELDS = ("early_epoch", "fraction", "upweight", "batch_size", "num_workers",
                   "image_size", "learning_rate", "weight_decay", "pretrained")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_relative_config(c):
    if c.get("methods") != list(RELATIVE_METHODS):
        raise ValueError("Run class_relative and class_matched_random together")
    if type(c.get("reference_epochs")) is not int or c["reference_epochs"] < 1:
        raise ValueError("reference_epochs must be positive")
    if type(c.get("epochs")) is not int or c["epochs"] < 1:
        raise ValueError("epochs must be positive")
    if type(c.get("pilot")) is not bool:
        raise ValueError("pilot must be explicit")
    if not c["pilot"] and c["epochs"] != c["reference_epochs"]:
        raise ValueError("The main comparison must match the ERM epoch budget")
    if type(c.get("control_seed_offset")) is not int or c["control_seed_offset"] < 0:
        raise ValueError("control_seed_offset must be nonnegative")
    # The selection window belongs to the reference ERM, not the shorter pilot.
    reference = dict(c, epochs=c["reference_epochs"], methods=list(METHODS))
    validate_config(reference)
    return c


def class_counts(scores, selected):
    labels = {r["sample_id"]:r["label"] for r in scores}
    if len(labels) != len(scores) or len(selected) != len(set(selected)):
        raise ValueError("Duplicate IDs in scores or selection")
    if not set(selected).issubset(labels):
        raise ValueError("Selected IDs are not training IDs")
    if any(label not in (0,1) for label in labels.values()):
        raise ValueError("Invalid bird label")
    counts = Counter(labels[i] for i in selected)
    return {str(label):counts[label] for label in (0,1)}


def weighted_class_totals(scores, selected, upweight):
    if not math.isfinite(upweight) or upweight < 1:
        raise ValueError("Invalid upweight")
    selected_counts = class_counts(scores, selected)
    totals = Counter(r["label"] for r in scores)
    return {str(label):totals[label]+(upweight-1)*selected_counts[str(label)] for label in (0,1)}


def make_plan(dynamics, train_ids, epoch, fraction, control_seed, upweight):
    """No subgroup metadata enter either selection rule."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    scores = score_examples(dynamics, train_ids, epoch)
    k = max(1, math.ceil(len(scores)*fraction))
    ordered = sorted(scores, key=lambda r:(-r["class_relative_loss"],r["sample_id"]))
    selected = [r["sample_id"] for r in ordered[:k]]
    counts = class_counts(scores, selected)
    generator = random.Random(control_seed)
    control = []
    for label in (0,1):
        pool = sorted(r["sample_id"] for r in scores if r["label"] == label)
        control.extend(generator.sample(pool,counts[str(label)]))
    control.sort()
    selections = dict(zip(RELATIVE_METHODS,(selected,control)))
    if class_counts(scores,control) != counts:
        raise RuntimeError("Class-matched random selection failed")
    masses = {method:weighted_class_totals(scores,ids,upweight) for method,ids in selections.items()}
    if masses[RELATIVE_METHODS[0]] != masses[RELATIVE_METHODS[1]]:
        raise RuntimeError("Expected class sampling masses differ")
    return dict(schema="class-relative-plan-v1", epoch=epoch, fraction=fraction,
                control_seed=control_seed, upweight=upweight, selected_count=k,
                train_ids=sorted(map(int,train_ids)), selected=selections,
                class_counts=counts, weighted_class_totals=masses,
                scores_sha256=digest_json(scores), subgroup_metadata_used=False)


def validate_selections(plan, actual):
    if set(actual) != set(RELATIVE_METHODS):
        raise ValueError("Missing or unexpected retraining method")
    for method in RELATIVE_METHODS:
        if actual[method] != plan["selected"][method]:
            raise ValueError(f"Selection differs from the fixed plan: {method}")


def load_reference(reference_root, seed, c):
    validate_relative_config(c)
    if seed not in c["seeds"]:
        raise ValueError("Seed not in config")
    folder = Path(reference_root)/f"seed_{seed}"
    if not (folder/"erm/COMPLETE").is_file():
        raise ValueError("Reference ERM is incomplete")
    metadata = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    identity = metadata["identity"]
    reference_config = validate_config(identity["config"])
    if identity["seed"] != seed or seed not in reference_config["seeds"]:
        raise ValueError("Reference seed mismatch")
    for field in TRAINING_FIELDS:
        if reference_config[field] != c[field]:
            raise ValueError(f"Reference setting differs: {field}")
    if reference_config["epochs"] != c["reference_epochs"]:
        raise ValueError("Reference epoch budget mismatch")
    result = json.loads((folder/"erm/result.json").read_text(encoding="utf-8"))
    if (result["signature"] != digest_json(identity) or result["seed"] != seed
            or result["method"] != "erm" or result["config"] != reference_config):
        raise ValueError("Reference ERM provenance mismatch")
    history = read_csv(folder/"erm/epoch_metrics.csv")
    if len(history) != c["reference_epochs"] or int(best_epoch(history)["epoch"]) != result["selected_epoch"]:
        raise ValueError("Reference checkpoint history is incomplete or inconsistent")
    dynamics = read_csv(folder/"erm/training_dynamics.csv")
    if any(int(r["seed"]) != seed for r in dynamics):
        raise ValueError("Reference dynamics seed mismatch")
    train_ids = sorted({int(r["sample_id"]) for r in dynamics})
    if len(train_ids) != identity["dataset"]["n_train"]:
        raise ValueError("Reference training IDs incomplete")
    early_scores(dynamics,train_ids,c["reference_epochs"])
    plan = make_plan(dynamics,train_ids,c["early_epoch"],c["fraction"],
                     seed+c["control_seed_offset"],c["upweight"])
    plan["reference_inputs_sha256"] = {name:file_sha(folder/name) for name in
        ("metadata.json","erm/result.json","erm/epoch_metrics.csv","erm/training_dynamics.csv","erm/COMPLETE")}
    plan["reference_initial_model_sha256"] = result["initial_model_sha256"]
    return dict(plan=plan,metadata=metadata,result=result,folder=folder)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Inspect class-matched selections without training or GPUs")
    parser.add_argument("reference")
    parser.add_argument("--config",default="configs/relative.json")
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    plan=load_reference(args.reference,args.seed,config)["plan"]
    output=Path(args.output)
    if output.exists():
        raise ValueError("Use a fresh selection-plan output path")
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,plan)
    print(f"seed={args.seed}: selected {plan['selected_count']} per method; "
          f"landbirds={plan['class_counts']['0']}, waterbirds={plan['class_counts']['1']}")
    print("Class counts and expected weighted sampling masses match exactly.")
