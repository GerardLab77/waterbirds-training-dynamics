"""Retrain from a fixed class-relative selection versus a class-matched random control."""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import train as trainer
from relative_rules import RELATIVE_METHODS, digest_json, load_reference, validate_relative_config
from waterbirds_core import file_sha, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/relative.json")
    parser.add_argument("--reference-root",default="runs/comparison-v1")
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--data-root",type=Path,default=Path("data"))
    parser.add_argument("--weights",type=Path,default=Path("weights/resnet18-f37072fd.pth"))
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after-epoch",type=int)
    args=parser.parse_args()
    c=validate_relative_config(json.loads(Path(args.config).read_text()))
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("stop-after-epoch must be positive")
    reference=load_reference(args.reference_root,args.seed,c)
    plan=reference["plan"]
    old_identity=reference["metadata"]["identity"]
    if args.synthetic != (old_identity["dataset"]["kind"]=="synthetic"):
        raise ValueError("Synthetic/real reference mismatch")
    if args.synthetic and c["pretrained"]:
        raise ValueError("Synthetic smoke must not claim pretrained training")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    import numpy as np
    trainer.torch,trainer.np=torch,np
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: request a GPU or choose --device cpu")
    torch.set_num_threads(max(1,min(4,int(os.environ.get("SLURM_CPUS_PER_TASK","4")))))
    train,dynamics,val,test,backgrounds,dataset_identity=trainer.datasets(args,c)
    if dataset_identity != old_identity["dataset"] or sorted(train.ids) != plan["train_ids"]:
        raise ValueError("Actual dataset/splits differ from reference ERM")
    versions={name:importlib.metadata.version(name) for name in ("torch","torchvision","numpy","wilds","Pillow")}
    device=torch.cuda.get_device_name(0) if args.device=="cuda" else "cpu"
    weights_hash=file_sha(args.weights) if c["pretrained"] else None
    for name,value in (("versions",versions),("python",platform.python_version()),
                       ("device",device),("weights_sha256",weights_hash)):
        if old_identity[name] != value:
            raise ValueError(f"Reference environment differs: {name}; use the existing environment")
    root=Path(__file__).resolve().parent
    identity=dict(experiment="class-relative-retraining-v1",config=c,seed=args.seed,
        dataset=dataset_identity,versions=versions,python=platform.python_version(),
        device=device,weights_sha256=weights_hash,selection_plan_sha256=digest_json(plan),
        reference_initial_model_sha256=plan["reference_initial_model_sha256"],
        source_hashes={name:file_sha(root/name) for name in
                      ("train_relative.py","relative_rules.py","train.py","detect.py","waterbirds_core.py")})
    seed_folder=Path(args.output)/f"seed_{args.seed}"
    seed_folder.mkdir(parents=True,exist_ok=True)
    metadata=seed_folder/"metadata.json"
    if metadata.exists():
        saved=json.loads(metadata.read_text())
        if not args.resume or saved["identity"] != identity:
            raise ValueError("Existing output requires matching --resume; use a new output for changed protocols")
    else:
        try:
            revision=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],
                stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            revision=None
        write_json(metadata,dict(identity=identity,git_revision=revision,
            reference_root=str(Path(args.reference_root).resolve()),
            slurm={k:v for k,v in os.environ.items() if k.startswith("SLURM_")}))
    write_json(seed_folder/"selection_plan.json",plan)
    print(f"seed={args.seed}: both selections contain {plan['selected_count']} images; "
          f"class counts={plan['class_counts']}",flush=True)
    for method in RELATIVE_METHODS:
        state=trainer.train_method(args,c,args.seed,method,seed_folder/method,
            (train,dynamics,val,test,backgrounds),identity,
            selected_ids=plan["selected"][method],
            expected_initial_hash=plan["reference_initial_model_sha256"])
        if state is None:
            print("Stopped at requested epoch; --resume will continue.",flush=True)
            return
    (seed_folder/"COMPLETE").write_text("Both class-matched retraining methods complete.\n",encoding="utf-8")
    print(f"Completed relative comparison: {seed_folder}",flush=True)


if __name__=="__main__":
    main()
