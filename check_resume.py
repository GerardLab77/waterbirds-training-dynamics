"""Compare a completed uninterrupted synthetic run with an interrupted/resumed run."""
import argparse
import json
from pathlib import Path

from waterbirds_core import METHODS, read_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uninterrupted")
    parser.add_argument("resumed")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--config",type=Path,help="Optional method list for a follow-up comparison")
    args=parser.parse_args()
    import torch
    methods=json.loads(args.config.read_text())["methods"] if args.config else METHODS
    for method in methods:
        a=Path(args.uninterrupted)/f"seed_{args.seed}"/method
        b=Path(args.resumed)/f"seed_{args.seed}"/method
        if not (a/"COMPLETE").is_file() or not (b/"COMPLETE").is_file():
            raise ValueError("Both runs must be complete")
        ca=torch.load(a/"last.pt",map_location="cpu",weights_only=False)
        cb=torch.load(b/"last.pt",map_location="cpu",weights_only=False)
        if ca["signature"]!=cb["signature"] or ca["history"]!=cb["history"]:
            raise ValueError("Environment/protocol or epoch metrics differ")
        if any(not torch.equal(ca["model"][k],cb["model"][k]) for k in ca["model"]):
            raise ValueError("Resumed model differs")
        for split in ("validation","test"):
            if read_csv(a/f"{split}_predictions.csv")!=read_csv(b/f"{split}_predictions.csv"):
                raise ValueError("Resumed predictions differ")
        print(f"{method}: identical final model, epoch metrics and predictions")


if __name__=="__main__":
    main()
