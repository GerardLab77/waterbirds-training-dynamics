"""Matched ERM/JTT/early-loss/random comparison; training requires an explicit CLI call."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

from waterbirds_core import (METHODS, early_scores, file_sha, group_metrics,
                             sample_weights, select_examples, validate_config,
                             write_csv, write_json)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class IndexedDataset:
    """Discard subgroup metadata at the boundary of every training/evaluation loader."""
    def __init__(self, subset):
        self.subset = subset
        self.ids = list(map(int, subset.indices))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image, label, _metadata = self.subset[index]
        return image, int(label), self.ids[index]


class SyntheticDataset:
    """Tiny deterministic images for plumbing checks, never scientific results."""
    def __init__(self, start, n, size):
        self.ids = list(range(start, start+n))
        self.size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        import torch
        sample = self.ids[index]
        generator = torch.Generator().manual_seed(sample + 1000)
        image = torch.rand((3, self.size, self.size), generator=generator)
        label = sample % 2
        image[label] += .2
        return image, label, sample


def datasets(args, c):
    if args.synthetic:
        train = SyntheticDataset(0, 32, c["image_size"])
        val, test = SyntheticDataset(100, 16, c["image_size"]), SyntheticDataset(200, 16, c["image_size"])
        backgrounds = {i: (i // 2) % 2 for i in train.ids + val.ids + test.ids}
        return train, train, val, test, backgrounds, dict(kind="synthetic", n_train=32, n_val=16, n_test=16)
    from torchvision import transforms
    from wilds import get_dataset
    dataset = get_dataset(dataset="waterbirds", root_dir=str(args.data_root), download=False)
    if dataset.n_classes != 2 or not {"background", "y"}.issubset(dataset.metadata_fields):
        raise ValueError("Unexpected Waterbirds schema")
    normalise = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    train_transform = transforms.Compose([transforms.Resize((c["image_size"], c["image_size"])),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalise])
    eval_transform = transforms.Compose([transforms.Resize((c["image_size"], c["image_size"])),
                                         transforms.ToTensor(), normalise])
    train = IndexedDataset(dataset.get_subset("train", transform=train_transform))
    dynamics = IndexedDataset(dataset.get_subset("train", transform=eval_transform))
    val = IndexedDataset(dataset.get_subset("val", transform=eval_transform))
    test = IndexedDataset(dataset.get_subset("test", transform=eval_transform))
    if train.ids != dynamics.ids or any(set(a.ids) & set(b.ids) for a, b in ((train,val),(train,test),(val,test))):
        raise ValueError("Overlapping or inconsistent split IDs")
    bg = dataset.metadata_fields.index("background")
    backgrounds = {i: int(dataset.metadata_array[i, bg]) for i in train.ids + val.ids + test.ids}
    identity = dict(kind="waterbirds", version=str(dataset.version), n_train=len(train),
                    n_val=len(val), n_test=len(test),
                    metadata_sha256=file_sha(Path(dataset.data_dir) / "metadata.csv"),
                    split_ids_sha256=hashlib.sha256(json.dumps([train.ids, val.ids, test.ids]).encode()).hexdigest())
    return train, dynamics, val, test, backgrounds, identity


def build_model(args, c):
    if args.synthetic:
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*c["image_size"]**2, 2))
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    if c["pretrained"]:
        # Only a local prepared file; no network access from training jobs.
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


def worker_seed(_worker):
    import torch
    import numpy as np
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, c, device, generator, weights=None, training=False):
    sampler = None
    if weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset), replacement=True,
                                                        generator=generator)
    return torch.utils.data.DataLoader(dataset, batch_size=c["batch_size"],
        shuffle=training and sampler is None, sampler=sampler, generator=generator,
        num_workers=c["num_workers"], pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed, persistent_workers=False)


def predict(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for images, labels, ids in loader:
            logits = model(images.to(device))
            labels = labels.to(device)
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            probabilities = logits.softmax(1)
            predictions = probabilities.argmax(1)
            for sample, label, prediction, loss, confidence in zip(ids.tolist(), labels.tolist(),
                    predictions.tolist(), losses.tolist(), probabilities.max(1).values.tolist()):
                if not np.isfinite(loss) or not np.isfinite(confidence):
                    raise RuntimeError("Non-finite prediction")
                rows.append(dict(sample_id=sample, label=label, prediction=prediction,
                                 correct=int(label == prediction), per_example_loss=loss,
                                 max_confidence=confidence))
    if len(rows) != len(loader.dataset) or len({r["sample_id"] for r in rows}) != len(rows):
        raise RuntimeError("Prediction IDs incomplete or duplicated")
    return rows


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_method(args, c, seed, method, folder, data, identity, scores=None):
    train, dynamics, val, test, backgrounds = data
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    last = folder / "last.pt"
    if folder.exists() and not args.resume:
        raise ValueError(f"Refusing to overwrite {folder}; use --resume or a new output directory")
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    selected = [] if method == "erm" else select_examples(scores, method, c["fraction"], seed)
    weights = None if method == "erm" or not selected else sample_weights(train.ids, selected, c["upweight"])
    write_json(folder / "selection.json", dict(method=method, selected_ids=selected,
        early_epoch=c["early_epoch"], fraction=c["fraction"], upweight=c["upweight"],
        rule="fixed early-epoch errors" if method == "jtt" else method,
        subgroup_metadata_used=False))
    generator = torch.Generator().manual_seed(seed)
    eval_generator = torch.Generator().manual_seed(seed+1)
    device = torch.device(args.device)
    model = build_model(args, c).to(device)
    initial_hash = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                                           for v in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    train_loader = make_loader(train, c, device, generator, weights, training=True)
    val_loader = make_loader(val, c, device, eval_generator)
    dynamics_loader = make_loader(dynamics, c, device, eval_generator)
    state = dict(epoch=0, history=[], dynamics=[], best_accuracy=-1, best_epoch=None,
                 best_model=None, signature=signature, initial_model_sha256=initial_hash,
                 method=method, selected_ids=selected)
    if last.exists():
        # Resume only checkpoints created by this project, not untrusted downloads.
        state = torch.load(last, map_location="cpu", weights_only=False)
        if (state["signature"] != signature or state["initial_model_sha256"] != initial_hash
                or state["method"] != method or state["selected_ids"] != selected):
            raise ValueError("Checkpoint config, sources, data, environment or initialisation changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["loader_rng"])
        eval_generator.set_state(state["eval_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif args.resume and any(folder.iterdir()) and (folder / "COMPLETE").exists():
        raise ValueError("Completed run is missing its checkpoint")
    if (folder / "COMPLETE").exists():
        return state
    started = time.perf_counter()
    previous_elapsed = state.get("elapsed_s", 0)
    for epoch in range(state["epoch"] + 1, c["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for images, labels, _ids in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        val_rows = predict(model, val_loader, device)
        val_accuracy = sum(r["correct"] for r in val_rows) / len(val_rows)
        state["history"].append(dict(epoch=epoch, training_loss=loss_sum/len(train),
                                     validation_accuracy=val_accuracy))
        if method == "erm":
            state["dynamics"].extend(dict(r, seed=seed, epoch=epoch)
                                     for r in predict(model, dynamics_loader, device))
        if val_accuracy > state["best_accuracy"]:
            state.update(best_accuracy=val_accuracy, best_epoch=epoch,
                         best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        state.update(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
                     eval_rng=eval_generator.get_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                     elapsed_s=previous_elapsed+time.perf_counter()-started)
        save_checkpoint(last, state)
        write_csv(folder / "epoch_metrics.csv", state["history"])
        if method == "erm":
            write_csv(folder / "training_dynamics.csv", state["dynamics"])
        print(f"seed={seed} method={method} epoch={epoch}/{c['epochs']} val={val_accuracy:.4f}", flush=True)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            return None
    # Test data are first evaluated here, after all checkpoint decisions.
    model.load_state_dict(state["best_model"])
    results = dict(seed=seed, method=method, signature=signature,
                   selected_epoch=state["best_epoch"], selected_count=len(selected),
                   initial_model_sha256=initial_hash, synthetic=args.synthetic,
                   config=c, dataset=identity["dataset"], training_elapsed_s=state["elapsed_s"])
    for split, subset in (("validation", val), ("test", test)):
        rows = predict(model, make_loader(subset, c, device, eval_generator), device)
        rows = [dict(r, background=backgrounds[r["sample_id"]]) for r in rows]
        write_csv(folder / f"{split}_predictions.csv", rows)
        results[split] = group_metrics(rows)
    write_json(folder / "result.json", results)
    (folder / "COMPLETE").write_text("Completed and evaluated selected checkpoint.\n", encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/comparison.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--weights", type=Path, default=Path("weights/resnet18-f37072fd.pth"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Debug interruption at a completed epoch")
    args = parser.parse_args()
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("stop-after-epoch must be positive")
    c = validate_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.seed not in c["seeds"]:
        raise ValueError("Seed not listed in this config")
    if args.synthetic and c["pretrained"]:
        raise ValueError("Synthetic smoke must not claim pretrained training")
    global torch, np
    import numpy as np
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: request a GPU or explicitly choose --device cpu")
    torch.set_num_threads(max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))))
    train, dynamics, val, test, backgrounds, dataset_identity = datasets(args, c)
    root = Path(__file__).resolve().parent
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "wilds", "Pillow")}
    identity = dict(config=c, seed=args.seed, dataset=dataset_identity,
                    versions=versions, python=platform.python_version(),
                    device=torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
                    weights_sha256=file_sha(args.weights) if c["pretrained"] and not args.synthetic else None,
                    source_hashes={name: file_sha(root / name) for name in ("train.py", "waterbirds_core.py")})
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    seed_folder = Path(args.output) / f"seed_{args.seed}"
    seed_folder.mkdir(parents=True, exist_ok=True)
    metadata = seed_folder / "metadata.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if not args.resume or saved["identity"] != identity:
            raise ValueError("Output exists; matching --resume required")
    else:
        write_json(metadata, dict(identity=identity, git_revision=revision,
            slurm={k: v for k,v in os.environ.items() if k.startswith("SLURM_")}))
    scores = None
    training_metadata = None
    for method in c["methods"]:
        state = train_method(args, c, args.seed, method, seed_folder/method,
                             (train, dynamics, val, test, backgrounds), identity, scores)
        if state is None:
            print("Stopped at requested epoch; --resume will continue this run.", flush=True)
            return
        if method == "erm":
            scores = early_scores(state["dynamics"], train.ids, c["early_epoch"])
            write_csv(seed_folder / "early_scores.csv", scores)
            labels = {r["sample_id"]: r["label"] for r in state["dynamics"] if r["epoch"] == 1}
            training_metadata = [dict(sample_id=i, label=labels[i], background=backgrounds[i])
                                 for i in sorted(train.ids)]
    # Retrospective metadata are exported only after every training method ends.
    write_csv(seed_folder / "training_metadata.csv", training_metadata)
    (seed_folder / "COMPLETE").write_text("All four methods complete.\n", encoding="utf-8")
    print(f"Completed comparison: {seed_folder}", flush=True)


if __name__ == "__main__":
    main()
