"""Explicit, HTTPS-verified input preparation; never runs during training."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from waterbirds_core import file_sha, write_json

DATA_URL = ("https://huggingface.co/datasets/pangwei/wilds-waterbirds/resolve/"
            "9df26fc8b17c3e54650b0ee634ab2d7804c638f1/waterbirds_v1.0.tar.gz")
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def safe_extract(archive, target):
    target = Path(target).resolve()
    for member in archive:
        destination = (target / member.name).resolve()
        if not destination.is_relative_to(target):
            raise ValueError("Unsafe archive path")
        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
        else:
            raise ValueError("Links and special archive members are not supported")


def download(url, destination):
    # urllib uses certificate verification by default; no insecure fallback.
    request = urllib.request.Request(url, headers={"User-Agent": "waterbirds-training-dynamics"})
    with urllib.request.urlopen(request, timeout=120) as response, Path(destination).open("wb") as output:
        shutil.copyfileobj(response, output)


def prepare_dataset(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "waterbirds_v1.0"
    if destination.exists():
        if not (destination / "metadata.csv").is_file():
            raise ValueError("Incomplete dataset directory; inspect it before moving it aside")
        return destination
    with tempfile.TemporaryDirectory(prefix="waterbirds-", dir=root) as tmp:
        temporary = Path(tmp)
        archive = temporary / "download.tar.gz"
        print("Downloading Waterbirds from its pinned WILDS source URL", flush=True)
        download(DATA_URL, archive)
        digest = file_sha(archive)
        unpacked = temporary / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive, "r:gz") as stream:
            safe_extract(stream, unpacked)
        candidates = [unpacked, unpacked / "waterbirds_v1.0"]
        source = next((p for p in candidates if (p / "metadata.csv").is_file()), None)
        if source is None:
            raise ValueError("No expected metadata.csv in the archive")
        source.replace(destination)
    write_json(root / "download_manifest.json", dict(url=DATA_URL, archive_sha256=digest,
        metadata_sha256=file_sha(destination / "metadata.csv")))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    args = parser.parse_args()
    prepare_dataset(args.data_root)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    weights = args.weights_dir / "resnet18-f37072fd.pth"
    if not weights.exists():
        temporary = weights.with_suffix(".download")
        download(WEIGHTS_URL, temporary)
        if not file_sha(temporary).startswith("f37072fd"):
            raise ValueError("Pretrained-weight SHA256 prefix mismatch")
        temporary.replace(weights)
    elif not file_sha(weights).startswith("f37072fd"):
        raise ValueError("Existing pretrained-weight checksum mismatch")
    from wilds import get_dataset
    dataset = get_dataset(dataset="waterbirds", root_dir=str(args.data_root), download=False)
    print(f"Prepared {len(dataset)} images and verified pretrained weights. No training performed.")


if __name__ == "__main__":
    main()
