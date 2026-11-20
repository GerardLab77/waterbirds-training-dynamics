"""Validate class-relative retraining and report its paired class-matched control."""
import argparse
import json
import statistics
from pathlib import Path

from relative_rules import (RELATIVE_METHODS, digest_json, load_reference,
                            validate_relative_config, validate_selections)
from waterbirds_core import best_epoch, file_sha, group_metrics, read_csv, write_csv, write_json


def read_predictions(folder, split, result, n):
    rows=read_csv(folder/f"{split}_predictions.csv")
    if len(rows) != n:
        raise ValueError("Incomplete evaluation predictions")
    metrics=group_metrics(rows)
    if metrics != result[split]:
        raise ValueError("Reported metrics disagree with saved predictions")
    signature=sorted((int(r["sample_id"]),int(r["label"]),int(r["background"])) for r in rows)
    return metrics,signature


def summarise_relative(root,reference_root,c,allow_synthetic=False):
    validate_relative_config(c)
    root=Path(root)
    rows,paired=[],[]
    common=None
    for seed in c["seeds"]:
        reference=load_reference(reference_root,seed,c)
        plan=reference["plan"]
        reference_identity=reference["metadata"]["identity"]
        if reference_identity["dataset"]["kind"]=="synthetic" and not allow_synthetic:
            raise ValueError("Synthetic runs are plumbing checks, not scientific results")
        folder=root/f"seed_{seed}"
        if not (folder/"COMPLETE").is_file():
            raise ValueError(f"Incomplete relative seed {seed}")
        metadata=json.loads((folder/"metadata.json").read_text())
        identity=metadata["identity"]
        if (identity.get("experiment")!="class-relative-retraining-v1" or
                identity["config"]!=c or identity["seed"]!=seed or
                identity["dataset"]!=reference_identity["dataset"]):
            raise ValueError("Relative run/config/dataset identity mismatch")
        for key in ("versions","python","device","weights_sha256"):
            if identity[key]!=reference_identity[key]:
                raise ValueError("Relative/reference environment mismatch")
        shared={k:identity[k] for k in ("dataset","versions","python","device","weights_sha256","source_hashes")}
        if common is None:
            common=shared
        if common!=shared:
            raise ValueError("Different experiment environments across seeds")
        if (identity["selection_plan_sha256"]!=digest_json(plan) or
                json.loads((folder/"selection_plan.json").read_text())!=plan or
                identity["reference_initial_model_sha256"]!=plan["reference_initial_model_sha256"]):
            raise ValueError("Selection/reference provenance changed")
        selection_files={}
        for method in RELATIVE_METHODS:
            method_folder=folder/method
            if not (method_folder/"COMPLETE").is_file():
                raise ValueError(f"Incomplete method {method}")
            selection=json.loads((method_folder/"selection.json").read_text())
            if (selection["subgroup_metadata_used"] or selection["method"]!=method or
                    selection["early_epoch"]!=c["early_epoch"] or
                    selection["fraction"]!=c["fraction"] or selection["upweight"]!=c["upweight"]):
                raise ValueError("Selection settings disagree with protocol")
            selection_files[method]=selection["selected_ids"]
        validate_selections(plan,selection_files)

        reference_signatures={}
        reference_row=dict(seed=seed,method="reference_erm",selected_epoch=reference["result"]["selected_epoch"],selected_count=0)
        for split,nkey in (("validation","n_val"),("test","n_test")):
            metrics,signature=read_predictions(reference["folder"]/"erm",split,
                reference["result"],identity["dataset"][nkey])
            reference_signatures[split]=signature
            reference_row.update({split+"_"+k:v for k,v in metrics.items()})
        if reference_row["validation_overall_accuracy"] != float(best_epoch(read_csv(reference["folder"]/"erm/epoch_metrics.csv"))["validation_accuracy"]):
            raise ValueError("Reference selected validation metric differs from history")
        # Short pilots do not compare a two-epoch model with a 30-epoch baseline.
        if c["epochs"]==c["reference_epochs"]:
            rows.append(reference_row)
        method_rows={}
        for method in RELATIVE_METHODS:
            method_folder=folder/method
            result=json.loads((method_folder/"result.json").read_text())
            if (result["signature"]!=digest_json(identity) or result["seed"]!=seed or
                    result["method"]!=method or result["config"]!=c or
                    result["selected_count"]!=plan["selected_count"] or
                    result["initial_model_sha256"]!=plan["reference_initial_model_sha256"]):
                raise ValueError("Method provenance or initial model mismatch")
            history=read_csv(method_folder/"epoch_metrics.csv")
            if len(history)!=c["epochs"] or int(best_epoch(history)["epoch"])!=result["selected_epoch"]:
                raise ValueError("Incomplete history or incorrect checkpoint selection")
            row=dict(seed=seed,method=method,selected_epoch=result["selected_epoch"],selected_count=plan["selected_count"])
            for split,nkey in (("validation","n_val"),("test","n_test")):
                metrics,signature=read_predictions(method_folder,split,result,identity["dataset"][nkey])
                if signature!=reference_signatures[split]:
                    raise ValueError("Evaluation IDs/labels/groups differ from reference")
                if split=="validation" and metrics["overall_accuracy"]!=float(best_epoch(history)["validation_accuracy"]):
                    raise ValueError("Selected validation metric differs from history")
                row.update({split+"_"+k:v for k,v in metrics.items()})
            rows.append(row)
            method_rows[method]=row
        train_ids=set(plan["train_ids"])
        val_ids={r[0] for r in reference_signatures["validation"]}
        test_ids={r[0] for r in reference_signatures["test"]}
        if train_ids & (val_ids|test_ids) or val_ids & test_ids:
            raise ValueError("Train/validation/test overlap")
        scored,control=(method_rows[m] for m in RELATIVE_METHODS)
        paired.append(dict(seed=seed,
            worst_group_delta_pp=100*(scored["test_worst_group_accuracy"]-control["test_worst_group_accuracy"]),
            overall_delta_pp=100*(scored["test_overall_accuracy"]-control["test_overall_accuracy"]),
            landbird_count=plan["class_counts"]["0"],waterbird_count=plan["class_counts"]["1"]))
    summary=[]
    for method in ("reference_erm",)+RELATIVE_METHODS:
        values=[r for r in rows if r["method"]==method]
        if not values:
            continue
        row=dict(method=method,seeds=len(values))
        for metric in ("test_overall_accuracy","test_worst_group_accuracy"):
            data=[r[metric] for r in values]
            row[metric+"_mean"]=statistics.mean(data)
            row[metric+"_sample_std"]=statistics.stdev(data) if len(data)>1 else ""
        summary.append(row)
    return rows,summary,paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--reference-root",default="runs/comparison-v1")
    parser.add_argument("--config",default="configs/relative.json")
    parser.add_argument("--output",required=True)
    parser.add_argument("--allow-synthetic",action="store_true")
    args=parser.parse_args()
    c=json.loads(Path(args.config).read_text())
    output=Path(args.output)
    if output.exists():
        raise ValueError("Use a fresh report output directory")
    rows,summary,paired=summarise_relative(args.run,args.reference_root,c,args.allow_synthetic)
    output.mkdir(parents=True)
    for name,values in (("per_seed",rows),("summary",summary),("paired_control_differences",paired)):
        write_csv(output/f"{name}.csv",values)
    write_json(output/"analysis.json",dict(config=c,analysis_source_sha256=file_sha(__file__),
        primary_comparison="class_relative minus class_matched_random",
        primary_metric="test worst-group accuracy, paired percentage-point difference",
        paired_worst_group_delta_pp_mean=statistics.mean(r["worst_group_delta_pp"] for r in paired),
        warning="Exploratory follow-up; seeds reuse images; no new detector generalisation claim."))
    print(f"Validated {len(c['seeds'])*2} new method/seed results with exact class-matched selections")
    if c["pilot"]:
        print("PILOT/SMOKE: validate execution, not a final research conclusion")
    for row in summary:
        print(f"{row['method']:22} overall={row['test_overall_accuracy_mean']:.4f} "
              f"worst-group={row['test_worst_group_accuracy_mean']:.4f}")
    for row in paired:
        print(f"seed={row['seed']}: class-relative minus class-matched random "
              f"worst-group={row['worst_group_delta_pp']:+.2f} pp, overall={row['overall_delta_pp']:+.2f} pp")
    if not args.allow_synthetic:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        methods=[r["method"] for r in summary]
        for seed in c["seeds"]:
            values=[next(r["test_worst_group_accuracy"] for r in rows if r["seed"]==seed and r["method"]==m) for m in methods]
            ax.plot(methods,values,marker="o",label=f"seed {seed}")
        ax.set(title="Class-relative retraining and class-matched random control",
               ylabel="Worst-group test accuracy",ylim=(0,1))
        ax.tick_params(axis="x",labelsize=9)
        ax.grid(alpha=.2)
        ax.legend()
        fig.savefig(output/"comparison.png",dpi=150)
        plt.close(fig)


if __name__=="__main__":
    main()
